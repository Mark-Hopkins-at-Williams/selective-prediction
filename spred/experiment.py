import json
import spred.tasks.normals
from spred.tasks import *
from spred.hub import spred_hub
from spred.analytics import ResultDatabase
from spred.confidence import MaxProb, MaxNonabstainProb
from spred.decoder import validate_and_analyze
from spred.analytics import ExperimentResult
from spred.train import BasicTrainer


class Experiment:

    def __init__(self, config, task=None):
        self.config = config
        self.task = task
        task_config = self.config['task']
        task_config = {k: task_config[k] for k in task_config if k != 'name'}
        if self.task is None:
            self.task = spred_hub.get_task(config['task']['name'])(**task_config)
        self.train_loader = self.task.init_train_loader(config['bsz'])
        self.validation_loader = self.task.init_validation_loader(config['bsz'])
        self.test_loader = self.task.init_test_loader(config['bsz'])


    def n_trials(self):
        if 'n_trials' in self.config:
            return self.config['n_trials']
        else:
            return 1


    def init_trainer(self, conf_fn):
        return BasicTrainer(self.config, self.train_loader,
                            self.validation_loader, conf_fn=conf_fn)


    def run(self):
        training_conf_fn = MaxProb()
        if 'regularizer' in self.config:
            if self.config['regularizer']['name'] == 'dac':
                training_conf_fn = MaxNonabstainProb()
        trainer = self.init_trainer(training_conf_fn)
        model, training_result = trainer()
        if 'evaluation' in self.config and self.config['evaluation'] == 'validation':
            eval_loader = self.validation_loader
        else:
            eval_loader = self.test_loader
        eval_results = []
        for confidence_config in self.config['confidences']:
            conf_constructor = spred_hub.get_confidence_fn(confidence_config['name'])
            params = {k: confidence_config[k] for k in confidence_config if k != "name"}
            conf_fn = conf_constructor(**params)
            conf_fn.train(self.task.init_train_loader(self.config['bsz']), model)
            model.set_confidence_extractor(conf_fn)
            task_name = self.config['task']['name'] if 'task' in self.config else None
            result = validate_and_analyze(model, eval_loader, task_name=task_name)
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

