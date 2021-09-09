from spred.decoder import InterfaceADecoder, InterfaceBDecoder
from spred.loss import init_loss_fn
from spred.model import InterfaceAFeedforward, InterfaceBFeedforward
from spred.model import PretrainedTransformer
from spred.train import BasicTrainer, CalibratedTrainer
from spred.viz import Visualizer
from abc import ABC, abstractmethod


class TaskFactory(ABC):
    def __init__(self, config):
        self._decoder_lookup = {'simple': InterfaceADecoder,
                                'abstaining': InterfaceBDecoder,
                                'pretrained': InterfaceADecoder}
        self.config = config
        self.architecture = self.config['network']['architecture']
        self.train_loader = None
        self.validation_loader = None
        self.test_loader = None
        self.train_loader = self.train_loader_factory()
        self.validation_loader = self.validation_loader_factory()
        self.test_loader = self.test_loader_factory()

    @abstractmethod
    def train_loader_factory(self):
        ...

    @abstractmethod
    def validation_loader_factory(self):
        ...

    @abstractmethod
    def test_loader_factory(self):
        ...

    def input_size(self):
        return self.train_loader.input_size()

    def output_size(self):
        return self.train_loader.output_size()

    def num_epochs(self):
        return self.config['trainer']['n_epochs']

    def select_trainer(self):
        return (CalibratedTrainer if self.config['network']['confidence'] == 'calib'
                else BasicTrainer)

    def decoder_factory(self):
        return self._decoder_lookup[self.architecture]()

    def trainer_factory(self):
        train_loader = self.train_loader_factory()
        validation_loader = self.validation_loader_factory()
        test_loader = self.test_loader_factory()
        decoder = self.decoder_factory()
        n_epochs = self.num_epochs()
        visualizer = self.visualizer_factory()
        trainer_class = self.select_trainer()
        trainer = trainer_class(self.config, train_loader, validation_loader,
                                test_loader, decoder, n_epochs, visualizer)
        return trainer

    def loss_factory(self):
        return init_loss_fn(self.config)

    def visualizer_factory(self):
        return None
