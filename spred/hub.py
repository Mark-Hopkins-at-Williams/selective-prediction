from abc import ABC, abstractmethod


class Hub:
    def __init__(self):
        self.tasks = dict()
        self.confs = dict()

    def register_task(self, task_id, task_constructor):
        self.tasks[task_id] = task_constructor

    def get_task(self, task_id):
        return self.tasks[task_id] if task_id in self.tasks else None

    def register_confidence_fn(self, conf_id, conf_constructor):
        self.confs[conf_id] = conf_constructor

    def get_confidence_fn(self, conf_id):
        return self.confs[conf_id] if conf_id in self.confs else None


spred_hub = Hub()
