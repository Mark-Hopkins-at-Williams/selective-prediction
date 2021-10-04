from abc import ABC, abstractmethod


class TaskHub:
    def __init__(self):
        self.tasks = dict()

    def register(self, task_id, task_constructor):
        self.tasks[task_id] = task_constructor

    def get(self, task_id):
        return self.tasks[task_id] if task_id in self.tasks else None


task_hub = TaskHub()


class Task(ABC):
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
