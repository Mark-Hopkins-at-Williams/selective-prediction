import matplotlib.pyplot as plt
import torch
import numpy
from numpy.random import multivariate_normal
from numpy import diag
from random import shuffle
from spred.loader import Loader
from spred.viz import Visualizer
from spred.task import TaskFactory


def generate_samples(means, variances, noise_dim, num_samples):
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
            pts1a = generate_samples([-1.0, 0.0], [variance1, variance2], self.noise_dim, bsz//4)
            pts1b = generate_samples([1.0, 0.0], [variance1, variance2], self.noise_dim, bsz//4)
            pts2a = generate_samples([0.0, -1.0], [variance2, variance1], self.noise_dim, bsz//4)
            pts2b = generate_samples([0.0, 1.0], [variance2, variance1], self.noise_dim, bsz//4)
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

    def output_size(self):
        return 2


class NormalsTaskFactory(TaskFactory):

    def train_loader_factory(self):
        n_train_batches = self.config['task']['n_train_batches']
        bsz = self.config['bsz']
        noise_dim = self.config['task']['noise_dim']
        return NormalsLoader(n_train_batches, bsz, noise_dim)

    def validation_loader_factory(self):
        n_batches = self.config['task']['n_validation_batches']
        bsz = self.config['bsz']
        noise_dim = self.config['task']['noise_dim']
        return NormalsLoader(n_batches, bsz, noise_dim)

    def test_loader_factory(self):
        n_test_batches = self.config['task']['n_test_batches']
        bsz = self.config['bsz']
        noise_dim = self.config['task']['noise_dim']
        return NormalsLoader(n_test_batches, bsz, noise_dim)

    def visualizer_factory(self):
        return NormalsVisualizer(self.config['n_epochs'])


class NormalsVisualizer(Visualizer):

    def __init__(self, last_epoch):
        super().__init__()
        self.last_epoch = last_epoch

    def viz(self, val_loader):
        def x_coords(pts):
            return [pt[0] for pt in pts]
        def y_coords(pts):
            return [pt[1] for pt in pts]
        batch = next(iter(val_loader.restart()))
        val_instances = batch['inputs']
        labels = batch['labels']
        pairs = [tuple(list(val_instances[i].numpy())[:2])
                 for i in range(len(val_instances))]
        class0 = [(pair, result) for (pair, result) in zip(pairs, labels)
                  if result == 0]
        class1 = [(pair, result) for (pair, result) in zip(pairs, labels)
                  if result == 1]
        plt.scatter(x_coords([c for (c,_) in class0]), y_coords([c for (c,_) in class0]),
                    label="class A", color="red")
        plt.scatter(x_coords([c for (c,_) in class1]), y_coords([c for (c,_) in class1]),
                    label="class B", color="blue")
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    def visualize(self, epoch, val_loader, results):
        def x_coords(pts):
            return [pt[0] for pt in pts]
        def y_coords(pts):
            return [pt[1] for pt in pts]
        if False and epoch == self.last_epoch:
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
