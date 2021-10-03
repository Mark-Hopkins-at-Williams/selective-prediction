from abc import ABC, abstractmethod


class Task(ABC):
    def __init__(self, config):
        self.config = config
        self.train_loader = None
        self.validation_loader = None
        self.test_loader = None
        self.train_loader = self.init_train_loader()
        self.validation_loader = self.init_validation_loader()
        self.test_loader = self.init_test_loader()

    @abstractmethod
    def init_train_loader(self):
        ...

    @abstractmethod
    def init_validation_loader(self):
        ...

    @abstractmethod
    def init_test_loader(self):
        ...

    def output_size(self):
        return self.train_loader.num_labels()
