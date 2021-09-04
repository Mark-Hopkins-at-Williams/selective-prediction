from spred.task import TaskFactory
from spred.tasks.normals.loader import NormalsLoader
from spred.tasks.normals.viz import NormalsVisualizer


class NormalsTaskFactory(TaskFactory):

    def __init__(self, config):
        super().__init__(config)
        self.bsz = self.config['trainer']['bsz']
        self.architecture = self.config['network']['architecture']
        self.noise_dim = self.config['task']['noise_dim']

    def train_loader_factory(self):
        if self.train_loader is None:
            n_train_batches = self.config['task']['n_train_batches']
            bsz = self.config['trainer']['bsz']
            noise_dim = self.config['task']['noise_dim']
            return NormalsLoader(n_train_batches, bsz, noise_dim)
        else:
            return self.train_loader.restart()

    def val_loader_factory(self):
        if self.validation_loader is None:
            n_validation_batches = self.config['task']['n_validation_batches']
            bsz = self.config['trainer']['bsz']
            noise_dim = self.config['task']['noise_dim']
            return NormalsLoader(n_validation_batches, bsz, noise_dim)
        else:
            return self.validation_loader.restart()

    def visualizer_factory(self):
        return NormalsVisualizer(self.config['trainer']['n_epochs'])


