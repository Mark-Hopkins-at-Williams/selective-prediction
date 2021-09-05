from spred.task import TaskFactory
from spred.tasks.rte.loader import RteLoader
from transformers import get_scheduler

class RteTaskFactory(TaskFactory):

    def __init__(self, config):
        super().__init__(config)
        self.architecture = self.config['network']['architecture']

    def train_loader_factory(self):
        bsz = self.config['trainer']['bsz']
        tokenizer = self.config['network']['base_model']
        return RteLoader(bsz, split="train", tokenizer=tokenizer)

    def val_loader_factory(self):
        bsz = self.config['trainer']['bsz']
        tokenizer = self.config['network']['base_model']
        return RteLoader(bsz, split="validation", tokenizer=tokenizer)