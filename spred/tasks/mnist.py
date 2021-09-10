import torch
from spred.loader import Loader
import os
from os.path import join
from torchvision import datasets
from torchvision import transforms
from spred.task import TaskFactory
from spred.model import InterfaceAFeedforward, InterfaceBFeedforward


DATA_DIR = os.getenv('SPRED_DATA').strip()
MNIST_DIR = join(DATA_DIR, 'mnist')
MNIST_TRAIN_DIR = join(MNIST_DIR, 'train')
MNIST_TEST_DIR = join(MNIST_DIR, 'test')

def confuse_two(labels):
    labels = labels.clone()
    one_and_sevens = (labels == 1) + (labels == 7)
    one_seven_shape = labels[one_and_sevens].shape
    # change the second argument for different weights
    new_labels = torch.randint(0, 2, one_seven_shape) 
    new_labels[new_labels == 0] = 7    
    labels[one_and_sevens] = new_labels
    return labels


def confuse_all(labels):
    labels = confuse_two(labels)
    one_and_sevens = (labels == 2) + (labels == 3)
    one_seven_shape = labels[one_and_sevens].shape
    new_labels = torch.randint(0, 2, one_seven_shape)
    new_labels[new_labels > 0] = 3
    new_labels[new_labels == 0] = 2
    labels[one_and_sevens] = new_labels
    one_and_sevens = (labels == 4) + (labels == 5)
    one_seven_shape = labels[one_and_sevens].shape
    new_labels = torch.randint(0, 4, one_seven_shape)
    new_labels[new_labels > 0] = 4
    new_labels[new_labels == 0] = 5
    labels[one_and_sevens] = new_labels
    return labels        


confuser_lookup = {'two': confuse_two,
                   'all': confuse_all}


class MnistLoader(Loader):
    
    def __init__(self, dataset, bsz=64, shuffle=True, confuser=lambda x: x):
        super().__init__()
        self.dataset = dataset
        self.bsz = bsz
        self.shuffle = shuffle
        self.loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=bsz, 
                                                  shuffle=shuffle)
        self.confuser = confuser
        
    def __iter__(self):
        for images, labels in self.loader:
            images = images.view(images.shape[0], -1)
            labels = self.confuser(labels)
            yield {'inputs': images, 'labels': labels}

    def __len__(self):
        return len(self.loader)

    def input_size(self):
        return 784

    def output_size(self):
        return 10

    def restart(self):
        result = MnistLoader(self.dataset, self.bsz, self.shuffle, self.confuser)
        result.batches = self.batches
        return result


class ConfusedMnistLoader(MnistLoader):
    
    def __init__(self, dataset, bsz=64, confuser='all', shuffle=True):
        super().__init__(dataset, bsz, shuffle, confuser_lookup[confuser])
        

class MnistPairLoader(Loader):
    def __init__(self, dataset, bsz=64, shuffle=True, confuser=lambda x:x):
        super().__init__()
        self.bsz = bsz
        self.dataset = dataset
        self.single_img_loader1 = torch.utils.data.DataLoader(dataset, 
                                                              batch_size=bsz, 
                                                              shuffle=shuffle)
        self.single_img_loader2 = torch.utils.data.DataLoader(dataset, 
                                                              batch_size=bsz, 
                                                              shuffle=shuffle)
        assert(len(self.single_img_loader1) == len(self.single_img_loader2))
        self.confuser = confuser
    
    def __len__(self):
        return len(self.single_img_loader1)

    def __iter__(self):
        for ((imgs1, lbls1), (imgs2, lbls2)) in zip(self.single_img_loader1, 
                                                    self.single_img_loader2):
            lbls1 = self.confuser(lbls1)
            lbls2 = self.confuser(lbls2)
            imgs1 = imgs1.view(imgs1.shape[0], -1)
            imgs2 = imgs2.view(imgs2.shape[0], -1)
            yield imgs1, imgs2, lbls1, lbls2


class ConfusedMnistPairLoader(MnistPairLoader):
    def __init__(self, dataset, bsz=64, confuser='all', shuffle=True):
        super().__init__(dataset, bsz, shuffle, confuser_lookup[confuser])


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
