import torch.optim as optim
from spred.loss import CrossEntropyLoss, NLLLoss, AbstainingLoss
from spred.loss import ConfidenceLoss4, PairwiseConfidenceLoss, DACLoss
from spred.decoder import InterfaceADecoder, InterfaceBDecoder
from abc import ABC, abstractmethod


class TaskFactory(ABC):
    def __init__(self, config):
        self.criterion_lookup = {'crossentropy': CrossEntropyLoss,
                                 'nll': NLLLoss,
                                 'conf1': AbstainingLoss,
                                 'conf4': ConfidenceLoss4,
                                 'pairwise': PairwiseConfidenceLoss,
                                 'dac': DACLoss}
        self._decoder_lookup = {'simple': InterfaceADecoder,
                                'abstaining': InterfaceBDecoder}
        self.config = config
        self.architecture = self.config['network']['architecture']

    @abstractmethod
    def train_loader_factory(self):
        ...

    @abstractmethod
    def val_loader_factory(self):
        ...

    @abstractmethod
    def model_factory(self, data):
        ...

    @abstractmethod
    def select_trainer(self):
        ...

    def decoder_factory(self):
        return self._decoder_lookup[self.architecture]()

    def trainer_factory(self):
        train_loader = self.train_loader_factory()
        val_loader = self.val_loader_factory()
        decoder = self.decoder_factory()
        model = self.model_factory(train_loader)
        optimizer = self.optimizer_factory(model)
        scheduler = self.scheduler_factory(optimizer)
        loss = self.loss_factory()
        n_epochs = self.config['trainer']['n_epochs']
        trainer_class = self.select_trainer()
        trainer = trainer_class(self.config, loss, optimizer, train_loader,
                                val_loader, decoder, n_epochs, scheduler)
        return trainer, model

    def optimizer_factory(self, model):
        optim_constrs = {'sgd': optim.SGD}
        oconfig = self.config['trainer']['optimizer']
        optim_constr = optim_constrs[oconfig['name']]
        params = {k: v for k, v in oconfig.items() if k != 'name'}
        return optim_constr(model.parameters(), **params)

    def loss_factory(self):
        lconfig = self.config['trainer']['loss']
        params = {k: v for k, v in lconfig.items() if k != 'name'}
        return self.criterion_lookup[lconfig['name']](**params)

    def scheduler_factory(self, optimizer):
        if self.config['trainer']['loss']['name'] == 'dac':
            return optim.lr_scheduler.MultiStepLR(optimizer,
                                                  milestones=[60, 80, 120],
                                                  gamma=0.5)
        else:
            return None
