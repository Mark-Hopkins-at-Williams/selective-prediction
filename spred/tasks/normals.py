import matplotlib.pyplot as plt
import torch
from numpy.random import multivariate_normal
from numpy import diag
from random import shuffle
from spred.loader import Loader
from spred.viz import Visualizer
from spred.task import TaskFactory


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
            yield {'inputs': instances, 'labels': labels}

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


class NormalsVisualizer(Visualizer):

    def __init__(self, last_epoch):
        super().__init__()
        self.last_epoch = last_epoch

    def visualize(self, epoch, val_loader, results):
        def x_coords(pts):
            return [pt[0] for pt in pts]
        def y_coords(pts):
            return [pt[1] for pt in pts]
        if epoch == self.last_epoch:
            val_instances = next(iter(val_loader.restart()))['inputs']
            pairs = [tuple(list(val_instances[i].numpy())[:2])
                     for i in range(len(val_instances))]
            class0 = [(pair, result) for (pair, result) in zip(pairs, results[:64])
                      if result['gold'] == 0]
            class1 = [(pair, result) for (pair, result) in zip(pairs, results[:64])
                      if result['gold'] == 1]
            pairs0 = [pair for (pair, _) in class0]
            confs0 = [result['confidence'] for (_, result) in class0]
            pairs1 = [pair for (pair, _) in class1]
            confs1 = [result['confidence'] for (_, result) in class1]
            plt.title('Confidence Visualization')
            plt.scatter(x_coords(pairs0), y_coords(pairs0), c=confs0,
                        label="class A", cmap='Reds')
            plt.scatter(x_coords(pairs1), y_coords(pairs1), c=confs1,
                        label="class B", cmap='Blues')
            plt.legend()
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()


class NormalsTaskFactory(TaskFactory):

    def train_loader_factory(self):
        if self.train_loader is None:
            n_train_batches = self.config['task']['n_train_batches']
            bsz = self.config['trainer']['bsz']
            noise_dim = self.config['task']['noise_dim']
            return NormalsLoader(n_train_batches, bsz, noise_dim)
        else:
            return self.train_loader.restart()

    def validation_loader_factory(self):
        if self.validation_loader is None:
            n_batches = self.config['task']['n_validation_batches']
            bsz = self.config['trainer']['bsz']
            noise_dim = self.config['task']['noise_dim']
            return NormalsLoader(n_batches, bsz, noise_dim)
        else:
            return self.validation_loader.restart()

    def test_loader_factory(self):
        if self.test_loader is None:
            n_test_batches = self.config['task']['n_test_batches']
            bsz = self.config['trainer']['bsz']
            noise_dim = self.config['task']['noise_dim']
            return NormalsLoader(n_test_batches, bsz, noise_dim)
        else:
            return self.test_loader.restart()

    def visualizer_factory(self):
        return NormalsVisualizer(self.config['trainer']['n_epochs'])
