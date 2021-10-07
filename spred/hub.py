from abc import ABC, abstractmethod


class Hub:
    def __init__(self):
        self.tasks = dict()
        self.confs = dict()
        self.losses = dict()
        self.models = dict()

    def register_task(self, task_id, task_constructor):
        self.tasks[task_id] = task_constructor

    def get_task(self, task_id):
        return self.tasks[task_id] if task_id in self.tasks else None

    def register_confidence_fn(self, conf_id, conf_constructor):
        self.confs[conf_id] = conf_constructor

    def get_confidence_fn(self, conf_id):
        return self.confs[conf_id] if conf_id in self.confs else None

    def register_loss_fn(self, loss_id, loss_constructor):
        self.losses[loss_id] = loss_constructor

    def get_loss_fn(self, loss_id):
        return self.losses[loss_id] if loss_id in self.losses else None

    def register_model(self, model_id, model_constructor):
        self.models[model_id] = model_constructor

    def get_model(self, model_id):
        return self.models[model_id] if model_id in self.models else None


spred_hub = Hub()
