import json
from spred.tasks.mnist import MnistTaskFactory
from spred.tasks.normals import NormalsTaskFactory
from spred.tasks.cola import ColaTaskFactory
from spred.tasks.sst2 import Sst2TaskFactory
from spred.tasks.rte import RteTaskFactory
from spred.analytics import ResultDatabase

task_factories = {'mnist': MnistTaskFactory,
                  'normals': NormalsTaskFactory,
                  'cola': ColaTaskFactory,
                  'sst2': Sst2TaskFactory,
                  'rte': RteTaskFactory}


class Experiment:

    def __init__(self, config):
        self.config = config
        self.task = task_factories[config['task']['name']](config)

    def n_trials(self):
        if 'n_trials' in self.config:
            return self.config['n_trials']
        else:
            return 1

    def run(self):
        trainer = self.task.trainer_factory()
        _, result = trainer()
        return result

    @classmethod
    def from_json(cls, model_config_path, train_config_path):
        with open(model_config_path, 'r') as f:
            model_config = json.load(f)
        with open(train_config_path, 'r') as f:
            train_config = json.load(f)
        model_config.update(train_config)
        return cls(model_config)

