from spred.loss import init_loss_fn
from spred.model import PretrainedTransformer
from spred.train import BasicTrainer, PostcalibratedTrainer, CocalibratedTrainer
from spred.viz import Visualizer
from abc import ABC, abstractmethod


class TaskFactory(ABC):
    def __init__(self, config):
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

    def trainer_factory(self):
        def select_trainer():
            if self.config['confidence'] == 'postcalib':
                return PostcalibratedTrainer
            elif self.config['confidence'] == 'cocalib':
                return CocalibratedTrainer
            else:
                return BasicTrainer

        train_loader = self.train_loader_factory()
        validation_loader = self.validation_loader_factory()
        test_loader = self.test_loader_factory()
        visualizer = self.visualizer_factory()
        trainer_class = select_trainer()
        trainer = trainer_class(self.config, train_loader, validation_loader,
                                test_loader, visualizer)
        return trainer

    def visualizer_factory(self):
        return None
