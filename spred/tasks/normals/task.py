from spred.task import TaskFactory
from spred.tasks.normals.loader import NormalsLoader
from spred.tasks.normals.viz import NormalsVisualizer


class NormalsTaskFactory(TaskFactory):

    def __init__(self, config):
        super().__init__(config)
        self.bsz = self.config['trainer']['bsz']
        self.architecture = self.config['network']['architecture']
        self.noise_dim = self.config['task']['noise_dim']
        self.n_train_batches = 200
        self.n_val_batches = 100
        self.train_loader = NormalsLoader(self.n_train_batches, self.bsz, self.noise_dim)
        self.val_loader = NormalsLoader(self.n_val_batches, self.bsz, self.noise_dim)

    def train_loader_factory(self):
        return self.train_loader.restart()

    def val_loader_factory(self):
        return self.val_loader.restart()

    def input_size(self):
        return self.train_loader.input_size()

    def output_size(self):
        return self.train_loader.output_size()

    def visualizer_factory(self):
        return NormalsVisualizer(self.config['trainer']['n_epochs'])


