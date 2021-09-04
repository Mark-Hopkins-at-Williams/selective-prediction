import json
from spred.tasks.mnist.task import MnistTaskFactory
from spred.tasks.normals.task import NormalsTaskFactory
from spred.tasks.glue.task import ColaTaskFactory
from spred.analytics import ResultDatabase

task_factories = {'mnist': MnistTaskFactory,
                  'normals': NormalsTaskFactory,
                  'cola': ColaTaskFactory}


class Experiment:

    def __init__(self, config):
        self.config = config
        self.task_factory = task_factories[config['task']['name']](config)

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
            result = experiment.run()
            results.append(result)
        return ResultDatabase(results)
