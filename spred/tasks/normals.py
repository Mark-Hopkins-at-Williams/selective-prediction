import matplotlib.pyplot as plt
import torch
import numpy
from numpy.random import multivariate_normal
from numpy import diag
from random import shuffle
from spred.loader import Loader
from spred.viz import Visualizer
from spred.task import Task


def generate_samples(means, variances, num_samples, noise_dim=0):
    return multivariate_normal(means + [0.0]*noise_dim,
                               diag(variances + [1.0]*noise_dim),
                               size=num_samples)


class NormalsLoader(Loader):
    
    def __init__(self, num_batches, bsz, noise_dim):
        super().__init__()
        self.bsz = bsz
        self.num_batches = num_batches
        self.noise_dim = noise_dim
        self.batches = []
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        for _ in range(num_batches):
            variance1 = 0.2
            variance2 = 0.1
            pts1a = generate_samples([-1.0, 0.0], [variance1, variance2], bsz//4, self.noise_dim)
            pts1b = generate_samples([1.0, 0.0], [variance1, variance2], bsz//4, self.noise_dim)
            pts2a = generate_samples([0.0, -1.0], [variance2, variance1], bsz//4, self.noise_dim)
            pts2b = generate_samples([0.0, 1.0], [variance2, variance1], bsz//4, self.noise_dim)
            labeled = ([(pt, 0) for pt in pts1a] +
                       [(pt, 0) for pt in pts1b] +
                       [(pt, 1) for pt in pts2a] +
                       [(pt, 1) for pt in pts2b])
            shuffle(labeled)
            instances = torch.tensor(numpy.array([pair[0] for pair in labeled])).float()
            labels = torch.tensor([pair[1] for pair in labeled])
            self.batches.append((instances, labels))

    def __iter__(self):
        for instances, labels in self.batches:
            yield {'inputs': instances.to(self.device), 'labels': labels.to(self.device)}

    def __len__(self):
        return self.num_batches

    def num_labels(self):
        return 2


class NormalsTask(Task):

    def __init__(self, config):
        super().__init__(config)

    def init_train_loader(self):
        n_batches = self.config['task']['n_train_batches']
        bsz = self.config['bsz']
        noise_dim = self.config['task']['noise_dim']
        return NormalsLoader(n_batches, bsz, noise_dim)

    def init_validation_loader(self):
        n_batches = self.config['task']['n_validation_batches']
        bsz = self.config['bsz']
        noise_dim = self.config['task']['noise_dim']
        return NormalsLoader(n_batches, bsz, noise_dim)

    def init_test_loader(self):
        n_batches = self.config['task']['n_test_batches']
        bsz = self.config['bsz']
        noise_dim = self.config['task']['noise_dim']
        return NormalsLoader(n_batches, bsz, noise_dim)

