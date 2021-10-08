import torch
from spred.loader import Loader
import os
from os.path import join
from torchvision import datasets
from torchvision import transforms
from spred.task import Task
from spred.model import Feedforward
from spred.hub import spred_hub
import tempfile


def confuse_none(labels):
    return labels


def confuse_ones_with_sevens(labels):
    labels = labels.clone()
    one_and_sevens = (labels == 1) + (labels == 7)
    one_seven_shape = labels[one_and_sevens].shape
    # change the second argument for different weights
    new_labels = torch.randint(0, 2, one_seven_shape) 
    new_labels[new_labels == 0] = 7    
    labels[one_and_sevens] = new_labels
    return labels


confuser_lookup = {'no': confuse_none,
                   '1<>7': confuse_ones_with_sevens}


class MnistLoader(Loader):
    
    def __init__(self, dataset, bsz=64, shuffle=True, confuser="no"):
        super().__init__()
        self.dataset = dataset
        self.bsz = bsz
        self.shuffle = shuffle
        self.loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=bsz, 
                                                  shuffle=shuffle)
        self.confuser = confuser_lookup[confuser]
        
    def __iter__(self):
        for images, labels in self.loader:
            images = images.view(images.shape[0], -1)
            labels = self.confuser(labels)
            yield {'inputs': images, 'labels': labels}

    def __len__(self):
        return len(self.loader)

    def num_labels(self):
        return 10


class MnistTask(Task):

    def __init__(self, confuser="no"):
        super().__init__()
        self.confuser = confuser

    @staticmethod
    def get_mnist_train():
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])
        return datasets.MNIST(tempfile.gettempdir(), download=True,
                              train=True, transform=transform)

    @staticmethod
    def get_mnist_test():
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])
        return datasets.MNIST(tempfile.gettempdir(), download=True,
                              train=False, transform=transform)


    def init_train_loader(self, bsz):
        ds = MnistTask.get_mnist_train()
        train_ds = [ds[i] for i in range(30000)]
        return MnistLoader(train_ds, bsz, shuffle=True, confuser=self.confuser)

    def init_validation_loader(self, bsz):
        ds = MnistTask.get_mnist_train()
        validation_ds = [ds[i] for i in range(30000, 60000)]
        return MnistLoader(validation_ds, bsz, shuffle=True, confuser=self.confuser)

    def init_test_loader(self, bsz):
        test_ds = MnistTask.get_mnist_test()
        return MnistLoader(test_ds, bsz, shuffle=True, confuser=self.confuser)


spred_hub.register_task('mnist', MnistTask)
