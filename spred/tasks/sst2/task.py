from spred.task import TaskFactory
from spred.tasks.sst2.loader import Sst2Loader
from transformers import get_scheduler

class Sst2TaskFactory(TaskFactory):

    def __init__(self, config):
        super().__init__(config)
        self.architecture = self.config['network']['architecture']

    def train_loader_factory(self):
        bsz = self.config['trainer']['bsz']
        tokenizer = self.config['network']['base_model']
        return Sst2Loader(bsz, split="train", tokenizer=tokenizer)

    def val_loader_factory(self):
        bsz = self.config['trainer']['bsz']
        tokenizer = self.config['network']['base_model']
        return Sst2Loader(bsz, split="validation", tokenizer=tokenizer)