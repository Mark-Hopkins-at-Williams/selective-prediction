from abc import ABC, abstractmethod
from spred.analytics import Evaluator, ExperimentResult, EpochResult
from spred.confidence import lookup_confidence_extractor
import torch
from tqdm import tqdm
from spred.loader import CalibrationLoader
from spred.confidence import CalibratorConfidence, max_prob
from spred.model import Feedforward
from spred.model import PretrainedTransformer
from spred.decoder import Decoder
import torch.optim as optim
from transformers import AdamW
from transformers import get_scheduler
from spred.loss import init_loss_fn


class Trainer(ABC):

    def __init__(self, config, train_loader, validation_loader, test_loader,
                 visualizer, compute_conf):
        self.config = config
        self.include_abstain = (self.config['confidence'] in
                                ['max_non_abstain', 'inv_abstain'])
        self.optimizer = None
        self.scheduler = None
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.n_epochs =  self.config['n_epochs']
        self.decoder = self.init_decoder()
        self.visualizer = visualizer
        self.compute_conf = compute_conf
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
        try:
            confidence_fn = lookup_confidence_extractor(self.config['confidence'])
        except Exception:
            confidence_fn = None
        if architecture in {"simple"}:
            return model_constructor(
                input_size=self.train_loader.input_size(),  # FIX THIS API!
                hidden_sizes=(128, 64),
                output_size=output_size,
                confidence_extractor=confidence_fn,
                loss_f = init_loss_fn(self.config),
                include_abstain_output = self.include_abstain
            )
        else:
            return model_constructor(
                base_model=self.config['network']['base_model'],
                output_size=output_size,
                confidence_extractor=confidence_fn,
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
    
    def __init__(self, config, train_loader, validation_loader, test_loader,
                 visualizer=None, compute_conf=True):
       super().__init__(config, train_loader, validation_loader, test_loader,
                        visualizer, compute_conf)

    def __call__(self):
        print("Training with config:")
        print(self.config)
        model = self.init_model()
        self.init_optimizer_and_scheduler(model)
        model = model.to(self.device)
        epoch_results = []
        for e in range(1, self.n_epochs+1):
            batch_loss = self.epoch_step(model)
            eval_result = self.validate_and_analyze(model, e)
            epoch_result = EpochResult(e, batch_loss, eval_result)
            epoch_results.append(epoch_result)
            print(str(epoch_result))
        return model, ExperimentResult(self.config, epoch_results)

    def epoch_step(self, model):
        running_loss = 0.
        denom = 0
        for batch in tqdm(self.train_loader, total=len(self.train_loader)):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            model.train()
            model_out = model(batch, compute_conf=self.compute_conf)
            output, loss, conf = model_out['outputs'], model_out['loss'], model_out['confidences']
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
        if self.visualizer is not None:
            self.visualizer.visualize(epoch, self.validation_loader, results)
        eval_result = Evaluator(results, validation_loss).get_result()
        return eval_result


class CalibratedTrainer(Trainer):

    def __init__(self, config, train_loader, validation_loader, test_loader,
                 visualizer=None):
        super().__init__(config, train_loader, validation_loader, test_loader,
                         visualizer, compute_conf=False)
        self.base_trainer = BasicTrainer(config, train_loader, test_loader, test_loader,
                                         visualizer, compute_conf=False)

    def __call__(self):
        print("Training with config:")
        print(self.config)
        base_model = self.init_model()
        base_model = base_model.to(self.device)
        self.base_trainer.init_optimizer_and_scheduler(base_model)
        self.calib_trainer = BasicTrainer(self.config,
                                          CalibrationLoader(base_model, self.validation_loader),
                                          CalibrationLoader(base_model, self.test_loader),
                                          CalibrationLoader(base_model, self.test_loader),
                                          self.visualizer, compute_conf=False)
        calibration_model = self.init_model(output_size=2)
        calibration_model = calibration_model.to(self.device)
        self.calib_trainer.init_optimizer_and_scheduler(calibration_model)
        confidence_fn = CalibratorConfidence(calibration_model)
        base_model.confidence_extractor = confidence_fn
        return self.train(base_model, calibration_model)

    @abstractmethod
    def train(self, base_model, calibration_model):
        ...


class PostcalibratedTrainer(CalibratedTrainer):

    def train(self, base_model, calibration_model):
        epoch_results = []
        for e in range(1, self.n_epochs + 1):
            base_model.notify(e)
            base_batch_loss = self.base_trainer.epoch_step(base_model)
            eval_result = self.base_trainer.validate_and_analyze(base_model, e)
            epoch_results.append(EpochResult(e, base_batch_loss, eval_result))
            print("epoch {}:".format(e))
            print(str(eval_result))
        for e in range(1, self.n_epochs + 1):
            base_model.eval()
            calibration_model.notify(e)
            calib_batch_loss = self.calib_trainer.epoch_step(calibration_model)
            eval_result = self.base_trainer.validate_and_analyze(base_model, e)
            epoch_results.append(EpochResult(e, calib_batch_loss, eval_result))
            print("epoch {}:".format(e))
            print(str(eval_result))
        return base_model, ExperimentResult(self.config, epoch_results)


class CocalibratedTrainer(CalibratedTrainer):

    def train(self, base_model, calibration_model):
        epoch_results = []
        for e in range(1, self.n_epochs + 1):
            base_model.train()
            base_model.notify(e)
            base_batch_loss = self.base_trainer.epoch_step(base_model)
            base_model.eval()
            calibration_model.notify(e)
            self.calib_trainer.epoch_step(calibration_model)
            eval_result = self.base_trainer.validate_and_analyze(base_model, e)
            epoch_results.append(EpochResult(e, base_batch_loss, eval_result))
            print("epoch {}:".format(e))
            print(str(eval_result))
        return base_model, ExperimentResult(self.config, epoch_results)
