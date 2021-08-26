import os
from os.path import join
from torchvision import datasets
from torchvision import transforms
from spred.task import TaskFactory
from spred.mnist.model import InterfaceAFeedforward, InterfaceBFeedforward
from spred.mnist.train import SingleTrainer as MnistSingleTrainer
from spred.mnist.train import PairwiseTrainer as MnistPairwiseTrainer
from spred.mnist.loader import MnistLoader, ConfusedMnistLoader
from spred.mnist.loader import MnistPairLoader, ConfusedMnistPairLoader


DATA_DIR = os.getenv('SPRED_DATA').strip()
MNIST_DIR = join(DATA_DIR, 'mnist')
MNIST_TRAIN_DIR = join(MNIST_DIR, 'train')
MNIST_TEST_DIR = join(MNIST_DIR, 'test')


class MnistTaskFactory(TaskFactory):

    def __init__(self, config):
        super().__init__(config)
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5,), (0.5,))])
        self._model_lookup = {'simple': InterfaceAFeedforward,
                              'abstaining': InterfaceBFeedforward}
        self.confuse = self.config['task']['confuse']
        self.bsz = self.config['trainer']['bsz']
        self.architecture = self.config['network']['architecture']

    def train_loader_factory(self):
        style = "pairwise" if self.architecture == 'confident' else "single"
        ds = datasets.MNIST(MNIST_TRAIN_DIR, download=True,
                            train=True, transform=self.transform)
        if self.confuse:
            loader_init = ConfusedMnistLoader if style == 'single' else ConfusedMnistPairLoader
            loader = loader_init(ds, self.bsz, self.confuse, shuffle=True)
        else:
            loader_init = MnistLoader if style == 'single' else MnistPairLoader
            loader = loader_init(ds, self.bsz, shuffle=True)
        return loader

    def val_loader_factory(self):
        ds = datasets.MNIST(MNIST_TEST_DIR, download=True,
                            train=False, transform=self.transform)
        if self.confuse:
            loader = ConfusedMnistLoader(ds, self.bsz, self.confuse, shuffle=True)
        else:
            loader = MnistLoader(ds, self.bsz, shuffle=True)
        return loader

    def model_factory(self, data):
        model_constructor = self._model_lookup[self.architecture]
        return model_constructor(confidence_extractor=self.config['network']['confidence'])

    def select_trainer(self):
        style = "pairwise" if self.architecture == 'confident' else "single"
        return MnistPairwiseTrainer if style == "pairwise" else MnistSingleTrainer
