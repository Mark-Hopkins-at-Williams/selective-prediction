import os
from os.path import join
from torchvision import datasets
from torchvision import transforms
from spred.task import TaskFactory
from spred.model import InterfaceAFeedforward, InterfaceBFeedforward
from spred.tasks.mnist.loader import MnistLoader, ConfusedMnistLoader
from spred.tasks.mnist.loader import MnistPairLoader, ConfusedMnistPairLoader


DATA_DIR = os.getenv('SPRED_DATA').strip()
MNIST_DIR = join(DATA_DIR, 'mnist')
MNIST_TRAIN_DIR = join(MNIST_DIR, 'train')
MNIST_TEST_DIR = join(MNIST_DIR, 'test')


class MnistTaskFactory(TaskFactory):

    def __init__(self, config):
        super().__init__(config)
        self._model_lookup = {'simple': InterfaceAFeedforward,
                              'abstaining': InterfaceBFeedforward}
        self.architecture = self.config['network']['architecture']

    def train_loader_factory(self):
        confuse = self.config['task']['confuse']
        bsz = self.config['trainer']['bsz']
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])
        style = "pairwise" if self.architecture == 'confident' else "single"
        ds = datasets.MNIST(MNIST_TRAIN_DIR, download=True,
                            train=True, transform=transform)
        train_ds = [ds[i] for i in range(30000)]
        if confuse:
            loader_init = ConfusedMnistLoader if style == 'single' else ConfusedMnistPairLoader
            loader = loader_init(train_ds, bsz, confuse, shuffle=True)
        else:
            loader_init = MnistLoader if style == 'single' else MnistPairLoader
            loader = loader_init(train_ds, bsz, shuffle=True)
        return loader

    def validation_loader_factory(self):
        confuse = self.config['task']['confuse']
        bsz = self.config['trainer']['bsz']
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])
        style = "pairwise" if self.architecture == 'confident' else "single"
        ds = datasets.MNIST(MNIST_TRAIN_DIR, download=True,
                            train=True, transform=transform)
        validation_ds = [ds[i] for i in range(30000, 60000)]
        if confuse:
            loader_init = ConfusedMnistLoader if style == 'single' else ConfusedMnistPairLoader
            loader = loader_init(validation_ds, bsz, confuse, shuffle=True)
        else:
            loader_init = MnistLoader if style == 'single' else MnistPairLoader
            loader = loader_init(validation_ds, bsz, shuffle=True)
        return loader

    def test_loader_factory(self):
        confuse = self.config['task']['confuse']
        bsz = self.config['trainer']['bsz']
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])
        ds = datasets.MNIST(MNIST_TEST_DIR, download=True,
                            train=False, transform=transform)
        if confuse:
            loader = ConfusedMnistLoader(ds, bsz, confuse, shuffle=True)
        else:
            loader = MnistLoader(ds, bsz, shuffle=True)
        return loader

    def input_size(self):
        return 784

    def output_size(self):
        return 10
