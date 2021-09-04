from spred.task import TaskFactory
from spred.tasks.glue.loader import ColaLoader
from transformers import get_scheduler

class ColaTaskFactory(TaskFactory):

    def __init__(self, config):
        super().__init__(config)
        self.bsz = self.config['trainer']['bsz']
        self.architecture = self.config['network']['architecture']
        self.train_loader = ColaLoader(self.bsz, split="train")
        self.val_loader = ColaLoader(self.bsz, split="dev")

    def train_loader_factory(self):
        return self.train_loader.restart()

    def val_loader_factory(self):
        return self.val_loader.restart()

    def input_size(self):
        ...

    def output_size(self):
        ...

    def scheduler_factory(self, optimizer):
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=self.num_epochs() * len(self.train_loader)
        )
        return lr_scheduler
