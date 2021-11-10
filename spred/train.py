import time
from spred.analytics import ExperimentResult, EpochResult
from spred.evaluate import Evaluator
from copy import deepcopy
import torch
from tqdm import tqdm
from spred.loader import CalibrationLoader
from spred.model import init_model
from spred.decoder import Decoder
from spred.confidence import RandomConfidence
import torch.optim as optim
from transformers import AdamW
from transformers import get_scheduler
from spred.hub import spred_hub


class BasicTrainer:

    def __init__(self, config, train_loader, validation_loader, conf_fn):
        self.config = config
        self.optimizer = None
        self.scheduler = None
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.n_epochs = self.config['n_epochs']
        self.regularizer = None
        self.include_abstain = False
        if 'regularizer' in self.config:
            loss_constructor = spred_hub.get_loss_fn(config['regularizer']['name'])
            loss_config = config['regularizer']
            params = {k: loss_config[k] for k in loss_config if k != "name"}
            self.regularizer = loss_constructor(**params)
            self.n_epochs += self.regularizer.bonus_epochs(self.n_epochs)
            self.include_abstain = self.regularizer.include_abstain()
        self.decoder = self.init_decoder()
        self.conf_fn = conf_fn
        self.device = (torch.device("cuda") if torch.cuda.is_available()
                       else torch.device("cpu"))

    def init_decoder(self):
        return Decoder(self.include_abstain)

    def init_model(self):
        model = init_model(self.config['model'],
                           self.regularizer,
                           self.include_abstain)
        model.set_confidence_extractor(self.conf_fn)
        return model

    def init_optimizer_and_scheduler(self, model):
        def init_optimizer():
            optim_constrs = {'sgd': optim.SGD,
                             'adamw': AdamW}
            oconfig = self.config['optimizer']
            optim_constr = optim_constrs[oconfig['name']]
            params = {k: v for k, v in oconfig.items() if k != 'name'}
            self.optimizer = optim_constr(model.parameters(), **params)

        def init_scheduler():
            try:
                scheduler_name = self.config['scheduler']['name']
            except KeyError:
                scheduler_name = None
            if scheduler_name == 'dac':
                self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                milestones=[60, 80, 120],
                                                                gamma=0.5)
            elif scheduler_name == 'linear':
                lr_scheduler = get_scheduler(
                    "linear",
                    optimizer=self.optimizer,
                    num_warmup_steps=0,
                    num_training_steps=self.n_epochs * len(self.train_loader)
                )
                self.scheduler = lr_scheduler

        init_optimizer()
        init_scheduler()

    def __call__(self):
        start_time = time.time()
        print("Training with config:")
        print(self.config)
        model = self.init_model()
        model = model.to(self.device)
        self.init_optimizer_and_scheduler(model)
        epoch_results = []
        top_epoch, top_state_dict = None, None
        top_validation_score = float('-inf')
        es_criterion = "accuracy"
        if "early_stopping_criterion" in self.config:
            es_criterion = self.config['early_stopping_criterion']
        for e in range(1, self.n_epochs+1):
            print("Epoch {}:".format(e))
            model.notify(e)
            batch_loss = self.epoch_step(model)
            eval_result = self.validate_and_analyze(model)
            epoch_result = EpochResult(e, batch_loss, eval_result)
            epoch_results.append(epoch_result)
            if eval_result[es_criterion] > top_validation_score: # TODO: what about criteria where smaller=better?
                top_epoch, top_state_dict = e, deepcopy(model.state_dict())
                top_validation_score = eval_result[es_criterion]
        print("Best validation accuracy at epoch {}".format(top_epoch))
        top_model = self.init_model()
        top_model.load_state_dict(top_state_dict)
        top_model = top_model.to(self.device)
        elapsed_time = time.time() - start_time
        return top_model, epoch_results, elapsed_time

    def epoch_step(self, model):
        running_loss = 0.
        denom = 0
        for batch in tqdm(self.train_loader, total=len(self.train_loader)):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            model.train()
            model_out = model(batch)
            output, loss = model_out['outputs'], model_out['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad()
            running_loss += loss.item()
            denom += 1
        return running_loss / denom

    def validate_and_analyze(self, model):
        model.eval()
        results = list(self.decoder(model, self.validation_loader))
        validation_loss = self.decoder.get_loss()
        task_name = self.config['task']['name'] if 'task' in self.config else None
        eval_result = Evaluator(results, validation_loss, task_name).get_result()
        return eval_result

