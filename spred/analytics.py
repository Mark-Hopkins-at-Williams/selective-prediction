from sklearn import metrics
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
from matplotlib import rcParams
import json
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma', 'DejaVu Sans', 'Ubuntu Condensed']


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
    worst_case = kendall_tau_distance(confidences, reversed(sorted(labels)))
    return dist/worst_case


kendalltau = relativized_kendall_tau_distance


class Evaluator:

    def __init__(self, predictions, loss=None):
        self._loss = loss
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

    def plot_roc(self):
        fpr, tpr, auc = self.roc_curve()
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label='AUROC = %0.2f' % auc)
        plt.legend(loc='lower right')
        axes = plt.gca()
        axes.set_ylim([-0.05, 1.05])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

    def plot_pr(self):
        precision, recall, auc = self.pr_curve()
        plt.title('Precision-Recall')
        plt.plot(recall, precision, 'b', label='AUPR = %0.2f' % auc)
        plt.legend(loc='lower right')
        axes = plt.gca()
        axes.set_ylim([-0.05, 1.05])
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.show()

    def get_result(self):
        _, _, auroc = self.roc_curve()
        _, _, aupr = self.pr_curve()
        _, _, capacity = self.risk_coverage_curve()
        return EvaluationResult.from_dict(
            {'train_loss': self._loss,
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
    def __init__(self, train_loss, auroc, aupr, capacity, precision, coverage,
                 avg_err_conf, avg_crr_conf, kendall_tau):
        self.train_loss = train_loss
        self.auroc = auroc
        self.aupr = aupr
        self.capacity = capacity
        self.precision = precision
        self.coverage = coverage
        self.avg_err_conf = avg_err_conf
        self.avg_crr_conf = avg_crr_conf
        self.kendall_tau = kendall_tau

    def loss(self):
        return self.train_loss

    def __str__(self):
        d = self.as_dict()
        return '  ' + '\n  '.join(['{}: {}'.format(key, d[key]) for key in d])

    def as_dict(self):
        return {'train_loss': self.train_loss,
                'kendall_tau': self.kendall_tau,
                'avg_err_conf': self.avg_err_conf,
                'avg_crr_conf': self.avg_crr_conf,
                'auroc': self.auroc,
                'aupr': self.aupr,
                'capacity': self.capacity,
                'precision': self.precision,
                'coverage': self.coverage}

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


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
        validation_result = EvaluationResult.from_dict(d['validation_result'])
        return cls(d['epoch'], d['train_loss'], validation_result)


class ExperimentResult:

    def __init__(self, config, epoch_results):
        self.config = config
        self.epoch_results = epoch_results

    def as_dict(self):
        results_json = [result.as_dict() for result in self.epoch_results]
        return {'config': self.config,
                'results': results_json}

    @classmethod
    def from_dict(cls, d):
        epoch_results = [EpochResult.from_dict(result) for result in d['results']]
        return cls(d['config'], epoch_results)

    def show_training_dashboard(self):
        fig, (ax1, ax2) = plt.subplots(2, sharex='all')
        fig.suptitle('Training Dashboard', fontsize='18')
        indexed_results = [(i+1, r) for (i, r) in enumerate(self.epoch_results)]
        x_axis = [i for (i, _) in indexed_results]
        train_losses = [r.get_train_loss() for (_, r) in indexed_results]
        valid_losses = [r.validation_result.loss() for (_, r) in indexed_results]
        valid_aurocs = [r.validation_result.auroc for (_, r) in indexed_results]
        valid_ktaus = [r.validation_result.kendall_tau for (_, r) in indexed_results]
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

    @staticmethod
    def average_list_of_results(list_of_results):
        def sum_result_dicts(this, other):
            def elementwise_add(ls1, ls2):
                assert (len(ls1) == len(ls2))
                return [(ls1[i] + ls2[i]) for i in range(len(ls1))]

            def process_key(key):
                if key != 'prediction_by_class':
                    return this[key] + other[key]
                else:
                    return {key: elementwise_add(this[key], other[key])
                            for key in this.keys()}

            assert (this.keys() == other.keys())
            return {key: process_key(key) for key in this.keys()}

        def normalize_result_dict(d, divisor):
            def elementwise_div(ls):
                return [element / divisor for element in ls]

            def process_key(key):
                if key != 'prediction_by_class':
                    return d[key] / divisor
                else:
                    return {key: elementwise_div(d[key])
                            for key in d.keys()}

            return {key: process_key(key) for key in d.keys()}

        result_sum = reduce(sum_result_dicts, list_of_results)
        avg_result = normalize_result_dict(result_sum,
                                           len(list_of_results))
        return avg_result


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
