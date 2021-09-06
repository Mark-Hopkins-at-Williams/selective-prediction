import sys
from sklearn import metrics
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
from matplotlib import rcParams, cm
import json
from statistics import mean
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma', 'DejaVu Sans', 'Ubuntu Condensed']
from collections import defaultdict


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


def kendall_tau_distance(confidences, predictions):
    n_discordant = 0
    n_positives_so_far = 0
    for pred in harsh_sort(confidences, predictions):
        if pred == 1:
            n_positives_so_far += 1
        else:
            n_discordant += n_positives_so_far
    return n_discordant


def relativized_kendall_tau_distance(confidences, labels):
    dist = kendall_tau_distance(confidences, labels)
    worst_case = kendall_tau_distance(sorted(confidences), reversed(sorted(labels)))
    return dist/worst_case


kendalltau = relativized_kendall_tau_distance


class Evaluator:

    def __init__(self, predictions, validation_loss=None):
        self.validation_loss = validation_loss
        self.y_true = [int(pred['pred'] == pred['gold']) for pred in predictions]
        self.y_scores = [pred['confidence'] for pred in predictions]
        self.avg_err_conf = 0
        self.avg_crr_conf = 0
        self.n_error = 0
        self.n_correct = 0
        self.n_published = 0
        self.n_preds = len(predictions)
        self.ktau = kendalltau([pred['confidence'] for pred in predictions],
                               [int(pred['pred'] == pred['gold']) for pred in predictions])
        for result in predictions:
            prediction = result['pred']
            gold = result['gold']
            confidence = result['confidence']
            recommends_abstention = result['abstain']
            if not recommends_abstention:
                self.n_published += 1
                if prediction == gold:
                    self.avg_crr_conf += confidence
                    self.n_correct += 1
                else:
                    self.avg_err_conf += confidence
                    self.n_error += 1
        if len(self.y_true) == 0 or len(self.y_scores) == 0:
            self.fpr, self.tpr, self.auroc = None, None, None
            self.precision, self.recall, self.aupr = None, None, None
        else:
            self.fpr, self.tpr, _ = metrics.roc_curve(self.y_true, self.y_scores,
                                                      pos_label=1)
            self.auroc = metrics.auc(self.fpr, self.tpr)
            precision, recall, _ = metrics.precision_recall_curve(self.y_true,
                                                                  self.y_scores)
            self.precision = np.insert(precision, 0,
                                       self.num_correct() / self.num_predictions(),
                                       axis=0)
            self.recall = np.insert(recall, 0, 1.0, axis=0)
            self.aupr = metrics.auc(self.recall, self.precision)

    def loss(self):
        return self._loss

    def num_errors(self):
        return self.n_error

    def num_correct(self):
        return self.n_correct

    def num_published(self):
        return self.n_published

    def num_predictions(self):
        return self.n_preds

    def kendall_tau(self):
        return self.ktau

    def pr_curve(self):
        return self.precision, self.recall, self.aupr

    def roc_curve(self):
        return self.fpr, self.tpr, self.auroc

    def risk_coverage_curve(self):
        # TODO: this function currently plots *unconditional* error rate
        #  against coverage
        if len(self.y_true) == 0 or len(self.y_scores) == 0:
            return None, None, None
        precision, _, thresholds = metrics.precision_recall_curve(self.y_true,
                                                                  self.y_scores)
        y_scores = sorted(self.y_scores)
        coverage = []
        N = len(y_scores)
        j = 0
        for i, t in enumerate(thresholds):
            while j < len(y_scores) and y_scores[j] < t:
                j += 1
            coverage.append((N - j) / N)
        coverage += [0.]
        conditional_err = 1 - precision
        unconditional_err = conditional_err * coverage
        coverage = np.array(coverage)
        capacity = 1 - metrics.auc(coverage, unconditional_err)
        return coverage, unconditional_err, capacity

    def get_result(self):
        _, _, auroc = self.roc_curve()
        _, _, aupr = self.pr_curve()
        _, _, capacity = self.risk_coverage_curve()
        return EvaluationResult(
            {'validation_loss': self.validation_loss,
             'kendall_tau': self.ktau,
             'avg_err_conf': (self.avg_err_conf / self.n_error
                              if self.n_error > 0 else 0),
             'avg_crr_conf': (self.avg_crr_conf / self.n_correct
                              if self.n_correct > 0 else 0),
             'auroc': auroc,
             'aupr': aupr,
             'capacity': capacity,
             'precision': (self.n_correct / self.n_published
                           if self.n_published > 0 else 0),
             'coverage': (self.n_published / self.n_preds
                          if self.n_preds > 0 else 0)
             })


class EvaluationResult:
    def __init__(self, result_dict):
        self.result_dict = result_dict

    def loss(self):
        if 'train_loss' in self.result_dict:
            return self.result_dict['train_loss']
        else:
            return None

    def __str__(self):
        return json.dumps(self.as_dict(), indent=4, sort_keys=True)

    __repr__ = __str__

    def __getitem__(self, key):
        return self.result_dict[key]

    def as_dict(self):
        return self.result_dict

    def __eq__(self, other):
        return self.result_dict == other.result_dict

    @staticmethod
    def averaged(list_of_results):
        def sum_result_dicts(this, other):
            def process_key(key):
                return this[key] + other[key]
            assert (this.keys() == other.keys())
            return {key: process_key(key) for key in this.keys()}

        def normalize_result_dict(d, divisor):
            def process_key(key):
                return d[key] / divisor
            return {key: process_key(key) for key in d.keys()}

        list_of_result_dicts = [x.as_dict() for x in list_of_results]
        result_sum = reduce(sum_result_dicts, list_of_result_dicts)
        avg_result = normalize_result_dict(result_sum,
                                           len(list_of_result_dicts))
        return EvaluationResult(avg_result)


class EpochResult:

    def __init__(self, epoch, train_loss, validation_result):
        self.epoch = epoch
        self.train_loss = train_loss
        self.validation_result = validation_result

    def get_train_loss(self):
        return self.train_loss

    def as_dict(self):
        return {'epoch': self.epoch,
                'train_loss': self.get_train_loss(),
                'validation_result': self.validation_result.as_dict()}

    @classmethod
    def from_dict(cls, d):
        try:
            valid_d = d['validation_result'].as_dict()
        except AttributeError:
            valid_d = d['validation_result'] # TODO: CLEAN THIS HACK UP
        validation_result = EvaluationResult(valid_d)
        return cls(d['epoch'], d['train_loss'], validation_result)

    def __eq__(self, other):
        return (self.epoch == other.epoch and
                self.train_loss == other.train_loss and
                self.validation_result == other.validation_result)

    def __str__(self):
        return json.dumps(self.as_dict(), indent=4, sort_keys=True)

    __repr__ = __str__

    @staticmethod
    def averaged(list_of_results):
        assert len(list_of_results) > 0
        validation_results = [x.validation_result for x in list_of_results]
        avg_validation_results = EvaluationResult.averaged(validation_results)
        avg_epoch = mean([x.epoch for x in list_of_results])
        avg_train_loss = mean([x.get_train_loss() for x in list_of_results])
        d = {'epoch': avg_epoch, 'train_loss': avg_train_loss,
             'validation_result': avg_validation_results}
        return EpochResult.from_dict(d)


class ExperimentResult:

    def __init__(self, config, epoch_results):
        self.config = config
        self.epoch_results = epoch_results

    def canonical_config(self):
        return str(tuple(sorted(self.config.items()))) # TODO: FIX THIS HACK

    def as_dict(self):
        results_json = [result.as_dict() for result in self.epoch_results]
        return {'config': self.config,
                'results': results_json}

    @classmethod
    def from_dict(cls, d):
        epoch_results = [EpochResult.from_dict(result) for result in d['results']]
        return cls(d['config'], epoch_results)

    def __str__(self):
        return json.dumps(self.as_dict(), indent=4, sort_keys=True)

    __repr__ = __str__

    @staticmethod
    def group_by_config(list_of_results):
        groups = defaultdict(list)
        for exp_result in list_of_results:
            groups[exp_result.canonical_config()].append(exp_result)
        return dict(groups)


class ResultDatabase:
    def __init__(self, experiment_results):
        self.results = experiment_results

    def save(self, filename):
        jsonified = [r.as_dict() for r in self.results]
        with open(filename, 'w') as f:
            json.dump(jsonified, f, indent=4)

    @classmethod
    def load(cls, filename):
        with open(filename, 'r') as f:
            experiment_results = json.load(f)
        experiment_results = [ExperimentResult.from_dict(d)
                              for d in experiment_results]
        return cls(experiment_results)

    def averaged(self):
        list_of_results = self.results
        assert len(list_of_results) > 0
        config_groups = ExperimentResult.group_by_config(list_of_results)
        final_result = []
        for (_, exp_results) in config_groups.items():
            grouped_epoch_results = defaultdict(list)
            for exp_result in exp_results:
                for epoch_result in exp_result.epoch_results:
                    grouped_epoch_results[epoch_result.epoch].append(epoch_result)
            avg_epoch_results = []
            for (epoch, epoch_results) in grouped_epoch_results.items():
                avg_epoch_results.append((epoch, EpochResult.averaged(epoch_results)))
            avg_epoch_results = [r for (_, r) in sorted(avg_epoch_results)]
            avg_exp_result = ExperimentResult(exp_results[0].config,
                                              avg_epoch_results)
            final_result.append(avg_exp_result)
        return final_result


def show_training_dashboard(exp_result):
    fig, (ax1, ax2) = plt.subplots(2, sharex='all')
    fig.suptitle('Training Dashboard', fontsize='18')
    indexed_results = [(i+1, r) for (i, r) in enumerate(exp_result.epoch_results)]
    x_axis = [i for (i, _) in indexed_results]
    train_losses = [r.get_train_loss() for (_, r) in indexed_results]
    valid_losses = [r.validation_result['validation_loss'] for (_, r) in indexed_results]
    valid_aurocs = [r.validation_result['auroc'] for (_, r) in indexed_results]
    valid_ktaus = [r.validation_result['kendall_tau'] for (_, r) in indexed_results]
    ax1.plot(x_axis, train_losses, 'b', label='train loss')
    ax1.plot(x_axis, valid_losses, 'r', label='valid loss')
    ax1.set(ylabel='loss')
    ax2.plot(x_axis, valid_aurocs, 'g', label='valid auroc')
    ax2.plot(x_axis, valid_ktaus, 'orange', label='valid ktau')
    ax1.legend(loc='upper right')
    ax2.legend(loc='lower right')
    ax2.set(ylabel='metric')
    plt.xlabel('epoch')
    plt.show()


def plot_metric(exp_results, metric_name):
    colors = iter(['red', 'orange', 'yellow', 'green', 'blue']*20)
    fig, ax = plt.subplots()
    fig.suptitle('Training Dashboard', fontsize='18')
    for name, exp_result in exp_results:
        indexed_results = [(i+1, r) for (i, r) in enumerate(exp_result.epoch_results)]
        x_axis = [i for (i, _) in indexed_results]
        valid_ktaus = [r.validation_result[metric_name] for (_, r) in indexed_results]
        color = next(colors)
        ax.plot(x_axis, valid_ktaus, color, label=name)
    ax.set(ylabel=metric_name)
    ax.legend()
    plt.xlabel('epoch')
    plt.show()


def main(result_files):
    result_dbs = [(file, ResultDatabase.load(file)) for file in result_files]
    avg_results = [(file, result_db.averaged()[0]) for (file, result_db) in result_dbs]
    plot_metric(avg_results, 'kendall_tau')


if __name__ == '__main__':
    i = 1
    files = []
    while i < len(sys.argv):
        files.append(sys.argv[i])
        i += 1
    main(files)
