import json
from spred.tasks.mnist.task import MnistTaskFactory
from spred.tasks.normals.task import NormalsTaskFactory
from spred.tasks.glue.task import ColaTaskFactory
from spred.tasks.sst2.task import Sst2TaskFactory
from spred.tasks.rte.task import RteTaskFactory
from spred.analytics import ResultDatabase

task_factories = {'mnist': MnistTaskFactory,
                  'normals': NormalsTaskFactory,
                  'cola': ColaTaskFactory,
                  'sst2': Sst2TaskFactory,
                  'rte': RteTaskFactory}


class Experiment:

    def __init__(self, config):
        self.config = config
        self.task_factory = task_factories[config['task']['name']](config)

    def n_trials(self):
        if 'n_trials' in self.config:
            return self.config['n_trials']
        else:
            return 1

    def run(self):
        trainer, model = self.task_factory.trainer_factory()
        _, result = trainer(model)
        return result


class ExperimentSequence:
    def __init__(self, experiments):
        self.experiments = experiments
    
    @classmethod
    def from_json(cls, configs_path):
        with open(configs_path, 'r') as f:
            configs = json.load(f)
        experiments = []
        for config in configs:
            experiments.append(Experiment(config))
        return cls(experiments)

    def run(self):
        results = []
        for experiment in self.experiments:
            for _ in range(experiment.n_trials()):
                result = experiment.run()
                results.append(result)
        return ResultDatabase(results)
