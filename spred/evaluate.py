"""
```evaluate.py``` implements evaluation metrics based on an abstract class ```EvaluationStatistic```.
```EvaluationStatistic``` interacts with ```Trainer``` or ```Evaluator``` through ```notify``` and ```notify_batch```.
Predictions made by the model are passed to the ```EvaluationStatistic``` metric so that the ```EvaluationStatistic```
object can update the metric.

```EvaluationStatistic``` also defines ```__call__``` method that return relevant data for other usages.

"""

import os
import sys
import numpy as np
import pandas as pd
import json
from torch import tensor
from abc import ABC, abstractmethod
from sklearn import metrics
from statistics import mean, median
from collections import defaultdict
from datasets import load_metric
from spred.analytics import EvaluationResult



class EvaluationStatistic(ABC):
    """
    Abstract class for all evaluation metrics

    """
    def __init__(self):
        self.stat = 0

    def notify(self, prediction):
        ...

    def notify_batch(self, batch):
        for pred in batch:
            self.notify(pred)

    def __call__(self):
        return self.stat


class ErrorCount(EvaluationStatistic):
    """
    Number of errors in predictions

    """
    def notify(self, pred):
        if pred['pred'] != pred['gold']:
            self.stat += 1


class CorrectCount(EvaluationStatistic):
    """
    Number of correct predictions

    """
    def notify(self, pred):
        if pred['pred'] == pred['gold']:
            self.stat += 1


class AverageNonabstainProb(EvaluationStatistic):
    """
    Average non-abstention probability

    """

    def __init__(self):
        super().__init__()
        self.numerator = 0.0
        self.denominator = 0.0

    def notify(self, pred):
        self.denominator += 1
        if 'abstain' in pred and pred['abstain']:
            self.numerator += pred['non_abstain_prob']
        else:
            self.numerator += 1.0
        self.stat = self.numerator / self.denominator


class AverageErrorConfidence(EvaluationStatistic):
    """
    Average error confidence

    """
    def __init__(self):
        super().__init__()
        self.numerator = 0.0
        self.denominator = 0.0

    def notify(self, pred):
        if pred['pred'] != pred['gold']:
            self.numerator += pred['confidence']
            self.denominator += 1
        if self.denominator != 0:
            self.stat = self.numerator / self.denominator


class AverageCorrectConfidence(EvaluationStatistic):
    """
    Average confidence of correct predictions

    """
    def __init__(self):
        super().__init__()
        self.numerator = 0.0
        self.denominator = 0.0

    def notify(self, pred):
        if pred['pred'] == pred['gold']:
            self.numerator += pred['confidence']
            self.denominator += 1
        if self.denominator != 0:
            self.stat = self.numerator / self.denominator


class Accuracy(EvaluationStatistic):
    """
    Accuracy

    """
    def __init__(self):
        super().__init__()
        self.numerator = 0.0
        self.denominator = 0.0

    def notify(self, pred):
        self.denominator += 1
        if pred['pred'] == pred['gold']:
            self.numerator += 1
        self.stat = self.numerator / self.denominator


class GlueMetric(EvaluationStatistic):
    """
    GLUE score

    """
    def __init__(self, task_name):
        super().__init__()
        self.use_glue = (task_name in {'cola', 'sst-2', 'mrpc', 'qqp', 'mnli',
                                       'qnli', 'rte', 'wnli'})
        self.metric = None
        if self.use_glue:
            self.metric = load_metric("glue", task_name)
        self.preds = []
        self.labels = []

    def notify(self, pred):
        if self.use_glue:
            self.preds.append(pred['pred'])
            self.labels.append(pred['gold'])

    def __call__(self):
        if len(self.preds) == 0 or self.metric is None:
            return dict()
        else:
            return self.metric.compute(predictions=tensor(self.preds),
                                       references=tensor(self.labels))


class PRCurve(EvaluationStatistic):
    """
    Precision-Recall Curve

    """
    def __init__(self):
        super().__init__()
        self.confs = []
        self.labels = []

    def notify(self, pred):
        self.confs.append(pred['confidence'])
        self.labels.append(int(pred['pred'] == pred['gold']))

    def __call__(self):
        if len(self.confs) == 0:
            return None
        else:
            from sklearn.metrics import precision_recall_curve as pr_curve
            precision, recall, _ = pr_curve(self.labels, self.confs)
            aupr = metrics.auc(recall, precision)
            return precision, recall, aupr


class RocCurve(EvaluationStatistic):
    """
    Receiver-Operator Characterisitics Curve

    """
    def __init__(self):
        super().__init__()
        self.confs = []
        self.labels = []

    def notify(self, pred):
        self.confs.append(pred['confidence'])
        self.labels.append(int(pred['pred'] == pred['gold']))

    def __call__(self):
        if len(self.confs) == 0:
            return None
        else:
            fpr, tpr, _ = metrics.roc_curve(self.labels, self.confs,
                                                     pos_label=1)
            auroc = metrics.auc(fpr, tpr)
            return fpr, tpr, auroc


class RiskCoverageCurve(EvaluationStatistic):
    """
    Risk-Coverage Curve

    """
    def __init__(self):
        super().__init__()
        self.confs = []
        self.labels = []

    def notify(self, pred):
        self.confs.append(pred['confidence'])
        self.labels.append(int(pred['pred'] == pred['gold']))

    def __call__(self):
        if len(self.confs) == 0:
            return None
        precision, _, thresholds = metrics.precision_recall_curve(self.labels,
                                                                  self.confs)
        y_scores = sorted(self.confs)
        coverage = []
        n = len(y_scores)
        j = 0
        for t in thresholds:
            while j < n and y_scores[j] < t:
                j += 1
            coverage.append((n - j) / n)
        coverage = np.array(coverage + [0.])
        selective_risk = np.array(list(precision))
        selective_risk = 1 - selective_risk
        capacity = 1 - metrics.auc(coverage, selective_risk)
        return coverage, selective_risk, capacity


class KendallTau(EvaluationStatistic):
    """
    Kendall-Tau Distance

    """
    def __init__(self):
        super().__init__()
        self.confs = []
        self.labels = []

    def notify(self, pred):
        self.confs.append(pred['confidence'])
        self.labels.append(int(pred['pred'] == pred['gold']))

    def __call__(self):
        return KendallTau.relativized_kendall_tau_distance(self.confs, self.labels)

    @staticmethod
    def harsh_sort(confidences, predictions):
        sorted_pairs = sorted(zip(confidences, predictions))
        result = []
        prev_conf = None
        sublist = []
        for (conf, pred) in sorted_pairs:
            if prev_conf is None:
                prev_conf = conf
                sublist = [pred]
            elif conf > prev_conf:
                prev_conf = conf
                result += sublist
                sublist = [pred]
            else:
                sublist = [pred] + sublist
        return result + sublist

    @staticmethod
    def kendall_tau_distance(confidences, predictions):
        n_discordant = 0
        n_positives_so_far = 0
        for pred in KendallTau.harsh_sort(confidences, predictions):
            if pred == 1:
                n_positives_so_far += 1
            else:
                n_discordant += n_positives_so_far
        return n_discordant

    @staticmethod
    def relativized_kendall_tau_distance(confidences, labels):
        dist = KendallTau.kendall_tau_distance(confidences, labels)
        worst_case = KendallTau.kendall_tau_distance(sorted(confidences), reversed(sorted(labels)))
        return dist/worst_case


class Evaluator:
    """
    ```Evaluator``` compiles ```EvaluationStatistic```'s and initilize it with predictions.
    Statistics are computed accordingly. One can access those statistics through the ```__getitem__``` by their name,
    and return the statistics in dictionary form with the function ```get_result```. There are two more methods:
    - ```loss```: return the average prediction loss
    - ```num_predictions```: return the total number of predictions passed to the evaluator
    """
    def __init__(self, predictions, validation_loss=None, task_name=None):
        self.stat_map = {'n_errors': ErrorCount(),
                         'n_correct': CorrectCount(),
                         'avg_non_abstain': AverageNonabstainProb(),
                         'avg_crr_conf': AverageCorrectConfidence(),
                         'avg_err_conf': AverageErrorConfidence(),
                         'accuracy': Accuracy(),
                         'kendall_tau': KendallTau(),
                         'risk_coverage': RiskCoverageCurve(),
                         'pr_curve': PRCurve(),
                         'roc_curve': RocCurve()}
        self.glue_metrics = GlueMetric(task_name)
        self.validation_loss = validation_loss
        self.n_preds = len(predictions)
        for pred in predictions:
            for stat_name in self.stat_map:
                self.stat_map[stat_name].notify(pred)
                self.glue_metrics.notify(pred)
        self.results = self.glue_metrics()
        self.results['validation_loss'] = self.validation_loss
        for stat_name in self.stat_map:
            if stat_name == 'risk_coverage':
                _, _, capacity = self.stat_map[stat_name]()
                self.results['capacity'] = capacity
            elif stat_name == 'pr_curve':
                _, _, aupr = self.stat_map[stat_name]()
                self.results['aupr'] = aupr
            elif stat_name == 'roc_curve':
                _, _, auroc = self.stat_map[stat_name]()
                self.results['auroc'] = auroc
            else:
                self.results[stat_name] = self.stat_map[stat_name]()

    def __getitem__(self, key):
        return self.results[key]

    def loss(self):
        return self._loss

    def num_predictions(self):
        return self.n_preds

    def get_result(self):
        return EvaluationResult(self.results)

