import json
from spred.tasks.mnist import MnistTaskFactory
from spred.tasks.normals import NormalsTaskFactory
from spred.tasks.glue import GlueTaskFactory
from spred.tasks.sst2 import Sst2TaskFactory
from spred.tasks.rte import RteTaskFactory
from spred.analytics import ResultDatabase
from spred.confidence import init_confidence_extractor, random_confidence
from spred.decoder import validate_and_analyze
from spred.analytics import ExperimentResult

task_factories = {'mnist': MnistTaskFactory,
                  'normals': NormalsTaskFactory,
                  "cola": GlueTaskFactory,
                  "mnli": GlueTaskFactory,
                  "mrpc": GlueTaskFactory,
                  "qnli": GlueTaskFactory,
                  "qqp": GlueTaskFactory,
                  "rte": GlueTaskFactory,
                  "sst2": GlueTaskFactory,
                  "stsb": GlueTaskFactory,
                  "wnli": GlueTaskFactory}

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
        training_conf_fn = random_confidence
        if self.config['loss']['name'] == 'ereg':
            confidence_config = {'name': 'max_prob'}
            training_conf_fn = init_confidence_extractor(confidence_config, self.config,
                                                         self.task, None)
        trainer = self.task.trainer_factory(training_conf_fn)
        model, training_result = trainer()
        if 'evaluation' in self.config and self.config['evaluation'] == 'validation':
            eval_loader = self.task.validation_loader_factory()
        else:
            eval_loader = self.task.test_loader_factory()
        eval_results = []
        for confidence_config in self.config['confidences']:
            conf_fn = init_confidence_extractor(confidence_config, self.config,
                                                self.task, model)
            model.set_confidence_extractor(conf_fn)
            result = validate_and_analyze(model, eval_loader, task_name=self.config['task']['name'])
            eval_results.append(result)
            print(confidence_config)
            print(result)
        return ExperimentResult(self.config, training_result, eval_results)

    @classmethod
    def from_json(cls, model_config_path, train_config_path):
        with open(model_config_path, 'r') as f:
            model_config = json.load(f)
        with open(train_config_path, 'r') as f:
            train_config = json.load(f)
        model_config.update(train_config)
        return cls(model_config)

