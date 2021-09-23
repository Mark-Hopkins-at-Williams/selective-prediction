from abc import ABC, abstractmethod
from spred.analytics import ExperimentResult, EpochResult
from spred.evaluate import Evaluator
from copy import deepcopy
import torch
from tqdm import tqdm
from spred.loader import CalibrationLoader
from spred.model import Feedforward
from spred.model import PretrainedTransformer
from spred.decoder import Decoder
import torch.optim as optim
from transformers import AdamW
from transformers import get_scheduler
from spred.loss import init_loss_fn


class Trainer(ABC):

    def __init__(self, config, train_loader, validation_loader, conf_fn):
        self.config = config
        self.include_abstain = self.config['loss']['name'] in ['dac']
        self.optimizer = None
        self.scheduler = None
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.n_epochs =  self.config['n_epochs']
        self.decoder = self.init_decoder()
        self.conf_fn = conf_fn
        self.device = (torch.device("cuda") if torch.cuda.is_available()
                       else torch.device("cpu"))


    def init_decoder(self):
        return Decoder(self.include_abstain)

    def init_model(self, output_size=None):
        model_lookup = {'simple': Feedforward,
                        'pretrained': PretrainedTransformer}
        architecture = self.config['network']['architecture']
        if output_size is None:
            output_size = self.train_loader.output_size()
        model_constructor = model_lookup[architecture]
        if architecture in {"simple"}:
            return model_constructor(
                input_size=self.train_loader.input_size(),  # FIX THIS API!
                hidden_sizes=(128, 64),
                output_size=output_size,
                confidence_extractor=self.conf_fn,
                loss_f = init_loss_fn(self.config),
                include_abstain_output = self.include_abstain
            )
        else:
            return model_constructor(
                base_model=self.config['network']['base_model'],
                output_size=output_size,
                confidence_extractor=self.conf_fn,
                include_abstain_output = self.include_abstain
            )

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
                print("*** WARNING: NO SCHEDULER PROVIDED ***")
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


class BasicTrainer(Trainer):
    
    def __init__(self, config, train_loader, validation_loader, conf_fn):
       super().__init__(config, train_loader, validation_loader, conf_fn)

    def __call__(self):
        print("Training with config:")
        print(self.config)
        model = self.init_model()
        model = model.to(self.device)
        self.init_optimizer_and_scheduler(model)
        epoch_results = []
        top_epoch, top_state_dict = None, None
        top_validation_score = float('-inf')
        for e in range(1, self.n_epochs+1):
            batch_loss = self.epoch_step(model)
            eval_result = self.validate_and_analyze(model, e)
            epoch_result = EpochResult(e, batch_loss, eval_result)
            epoch_results.append(epoch_result)
            print(str(epoch_result))
            if eval_result['accuracy'] > top_validation_score:
                top_epoch, top_state_dict = e, deepcopy(model.state_dict())
                top_validation_score = eval_result['accuracy']
        print("Best validation accuracy at epoch {}".format(top_epoch))
        top_model = self.init_model()
        top_model.load_state_dict(top_state_dict)
        top_model = top_model.to(self.device)
        eval_result = self.validate_and_analyze(top_model, top_epoch)
        print(str(eval_result))
        return top_model, epoch_results

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

    def validate_and_analyze(self, model, epoch):
        model.eval()
        results = list(self.decoder(model, self.validation_loader))
        validation_loss = self.decoder.get_loss()
        eval_result = Evaluator(results, validation_loss,
                                task_name=self.config['task']['name']).get_result()
        return eval_result

