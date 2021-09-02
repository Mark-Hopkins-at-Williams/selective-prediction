import matplotlib.pyplot as plt
import torch
from numpy.random import multivariate_normal
from numpy import diag
from random import shuffle
from spred.loader import Loader


def x_coords(pts):
    return [pt[0] for pt in pts]


def y_coords(pts):
    return [pt[1] for pt in pts]


def sample_2d_normal(means, variances, num_samples):
    samples = multivariate_normal(means,
                                  diag(variances),
                                  size=num_samples)
    return samples


def generate_samples(means, variances, noise_dim, num_samples):
    return sample_2d_normal(means + [0.0]*noise_dim,
                            variances + [1.0]*noise_dim,
                            num_samples)


class NormalsLoader(Loader):
    
    def __init__(self, num_batches, bsz, noise_dim):
        super().__init__()
        self.bsz = bsz
        self.num_batches = num_batches
        self.noise_dim = noise_dim
        self.batches = []
        for _ in range(num_batches):
            variance1 = 0.5
            variance2 = 0.1
            pts1a = generate_samples([-1.0, 0.0], [variance1, variance2], self.noise_dim, bsz//4)
            pts1b = generate_samples([1.0, 0.0], [variance1, variance2], self.noise_dim, bsz//4)
            pts2a = generate_samples([0.0, -1.0], [variance2, variance1], self.noise_dim, bsz//4)
            pts2b = generate_samples([0.0, 1.0], [variance2, variance1], self.noise_dim, bsz//4)
            labeled = ([(pt, 0) for pt in pts1a] +
                       [(pt, 0) for pt in pts1b] +
                       [(pt, 1) for pt in pts2a] +
                       [(pt, 1) for pt in pts2b])
            shuffle(labeled)
            instances = torch.tensor([pair[0] for pair in labeled]).float()
            labels = torch.tensor([pair[1] for pair in labeled])
            self.batches.append((instances, labels))

    def __iter__(self):
        for instances, labels in self.batches:
            yield instances, labels

    def __len__(self):
        return self.num_batches

    def input_size(self):
        return self.noise_dim + 2

    def output_size(self):
        return 2

    def restart(self):
        result = NormalsLoader(self.num_batches, self.bsz, self.noise_dim)
        result.batches = self.batches
        return result
