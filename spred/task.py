from abc import ABC, abstractmethod


class Task(ABC):
    """
    The abstract class for tasks (e.g. Mnist, GLUE)
    Configures data loaders that is compatible for the task
    contains information for the size of the output logits (```output_size```)
    
    """
    def __init__(self):
        self.output_sz = None

    @abstractmethod
    def init_train_loader(self, bsz):
        ...

    @abstractmethod
    def init_validation_loader(self, bsz):
        ...

    @abstractmethod
    def init_test_loader(self, bsz):
        ...

    def output_size(self):
        if self.output_sz is None:
            self.output_sz = self.init_train_loader(1).num_labels()
        return self.output_sz
