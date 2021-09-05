import torch.optim as optim
from transformers import AdamW
from transformers import get_scheduler
from spred.decoder import InterfaceADecoder, InterfaceBDecoder
from spred.loss import CrossEntropyLoss, NLLLoss, AbstainingLoss
from spred.loss import PairwiseConfidenceLoss, DACLoss
from spred.loss import CrossEntropyLossWithErrorRegularization
from spred.model import InterfaceAFeedforward, InterfaceBFeedforward
from spred.model import PretrainedTransformer
from spred.train import SingleTrainer, PairwiseTrainer
from spred.viz import Visualizer
from abc import ABC, abstractmethod


class TaskFactory(ABC):
    def __init__(self, config):
        self.criterion_lookup = {'crossentropy': CrossEntropyLoss,
                                 'nll': NLLLoss,
                                 'conf1': AbstainingLoss,
                                 'pairwise': PairwiseConfidenceLoss,
                                 'dac': DACLoss,
                                 'ce_w_er': CrossEntropyLossWithErrorRegularization}
        self._decoder_lookup = {'simple': InterfaceADecoder,
                                'abstaining': InterfaceBDecoder,
                                'pretrained': InterfaceADecoder}
        self._model_lookup = {'simple': InterfaceAFeedforward,
                              'abstaining': InterfaceBFeedforward,
                              'pretrained': PretrainedTransformer}
        self.config = config
        self.architecture = self.config['network']['architecture']
        self.train_loader = None
        self.validation_loader = None
        self.train_loader = self.train_loader_factory()
        self.validation_loader = self.val_loader_factory()

    @abstractmethod
    def train_loader_factory(self):
        ...

    @abstractmethod
    def val_loader_factory(self):
        ...

    def input_size(self):
        return self.train_loader.input_size()

    def output_size(self):
        return self.train_loader.output_size()

    def num_epochs(self):
        return self.config['trainer']['n_epochs']

    def model_factory(self):
        model_constructor = self._model_lookup[self.architecture]
        if self.architecture == "simple":
            return model_constructor(
                input_size=self.input_size(),  # FIX THIS API!
                hidden_sizes=(128, 64),
                output_size=self.output_size(),
                confidence_extractor=self.config['network']['confidence'],
                loss_f = self.loss_factory()
            )
        else:
            return model_constructor(
                base_model=self.config['network']['base_model'],
                confidence_extractor=self.config['network']['confidence']
            )

    def select_trainer(self):
        style = "pairwise" if self.architecture == 'confident' else "single"
        return PairwiseTrainer if style == "pairwise" else SingleTrainer

    def decoder_factory(self):
        return self._decoder_lookup[self.architecture]()

    def trainer_factory(self):
        train_loader = self.train_loader_factory()
        val_loader = self.val_loader_factory()
        decoder = self.decoder_factory()
        model = self.model_factory()
        optimizer = self.optimizer_factory(model)
        scheduler = self.scheduler_factory(optimizer)
        loss = self.loss_factory()
        n_epochs = self.num_epochs()
        visualizer = self.visualizer_factory()
        trainer_class = self.select_trainer()
        trainer = trainer_class(self.config, loss, optimizer, train_loader,
                                val_loader, decoder, n_epochs, scheduler, visualizer)
        return trainer, model

    def optimizer_factory(self, model):
        optim_constrs = {'sgd': optim.SGD,
                         'adamw': AdamW}
        oconfig = self.config['trainer']['optimizer']
        optim_constr = optim_constrs[oconfig['name']]
        params = {k: v for k, v in oconfig.items() if k != 'name'}
        return optim_constr(model.parameters(), **params)

    def loss_factory(self):
        lconfig = self.config['trainer']['loss']
        params = {k: v for k, v in lconfig.items() if k != 'name'}
        return self.criterion_lookup[lconfig['name']](**params)

    def scheduler_factory(self, optimizer):
        try:
            scheduler_name = self.config['trainer']['scheduler']['name']
        except KeyError:
            print("*** WARNING: NO SCHEDULER PROVIDED ***")
            scheduler_name = None
        if scheduler_name == 'dac':
            return optim.lr_scheduler.MultiStepLR(optimizer,
                                                  milestones=[60, 80, 120],
                                                  gamma=0.5)
        elif scheduler_name == 'linear':
            lr_scheduler = get_scheduler(
                "linear",
                optimizer=optimizer,
                num_warmup_steps=0,
                num_training_steps=self.num_epochs() * len(self.train_loader)
            )
            return lr_scheduler

    def visualizer_factory(self):
        return None
