import os
import sys
from sklearn import metrics
import numpy as np
import pandas as pd
import seaborn as sns
from functools import reduce
import matplotlib.pyplot as plt
from matplotlib import rcParams, cm
import json
from statistics import mean
from collections import defaultdict
from statistics import median
from datasets import load_metric
from random import shuffle

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

    def record_elapsed_time(self, elapsed_time):
        self.result_dict['elapsed_time'] = elapsed_time

    def __eq__(self, other):
        return self.result_dict == other.result_dict

    @staticmethod
    def merge(list_of_results):
        list_of_result_dicts = [x.as_dict() for x in list_of_results]
        all_keys = set()
        for d in list_of_result_dicts:
            all_keys |= set(d.keys())
        merged = {}
        for key in all_keys:
            merged[key] = [x[key] for x in list_of_result_dicts]
        return merged

    @staticmethod
    def averaged(list_of_results):
        merged = EvaluationResult.merge(list_of_results)
        result = {key: mean(merged[key]) for key in merged}
        return EvaluationResult(result)

    @staticmethod
    def median(list_of_results):
        merged = EvaluationResult.merge(list_of_results)
        result = {key: median(merged[key]) for key in merged}
        return EvaluationResult(result)

    @staticmethod
    def all(list_of_results):
        merged = EvaluationResult.merge(list_of_results)
        result = {key: merged[key] for key in merged}
        return EvaluationResult(result)

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

    @staticmethod
    def median(list_of_results):
        assert len(list_of_results) > 0
        validation_results = [x.validation_result for x in list_of_results]
        avg_validation_results = EvaluationResult.median(validation_results)
        avg_epoch = median([x.epoch for x in list_of_results])
        avg_train_loss = median([x.get_train_loss() for x in list_of_results])
        d = {'epoch': avg_epoch, 'train_loss': avg_train_loss,
             'validation_result': avg_validation_results}
        return EpochResult.from_dict(d)

class ExperimentResult:

    def __init__(self, config, epoch_results, eval_results, training_time):
        self.config = config
        self.epoch_results = epoch_results
        self.eval_results = eval_results
        self.training_time = training_time

    def as_dict(self):
        epoch_results_json = [result.as_dict() for result in self.epoch_results]
        eval_results_json = [result.as_dict() for result in self.eval_results]
        return {'config': self.config,
                'training_results': epoch_results_json,
                'training_time': self.training_time,
                'confidence_results': eval_results_json}

    @classmethod
    def from_dict(cls, d):
        epoch_results = [EpochResult.from_dict(result) for result in d['training_results']]
        eval_results = [EvaluationResult(result) for result in d['confidence_results']]
        return cls(d['config'], epoch_results, eval_results, d['training_time'])

    def __str__(self):
        return json.dumps(self.as_dict(), indent=4, sort_keys=True)

    __repr__ = __str__


class ResultDatabase:
    def __init__(self, experiment_results):
        self.results = experiment_results

    def save(self, filename):
        jsonified = [r.as_dict() for r in self.results]
        with open(filename, 'w') as f:
            json.dump(jsonified, f, indent=4)

    @classmethod
    def load(cls, directory):
        files = [os.path.join(directory, f)
                 for f in os.listdir(directory)
                 if f.endswith('.results.json')]
        experiment_results = []
        for filename in files:
            with open(filename, 'r') as f:
                file_results = [ExperimentResult.from_dict(d)
                                for d in json.load(f)]
                experiment_results += file_results
        return cls(experiment_results)

    def summary(self, combine_epoch_fn=EpochResult.median, combine_eval_fn=EvaluationResult.all):
        exp_results = self.results
        grouped_epoch_results = defaultdict(list)
        grouped_conf_results = [[] for _ in range(len(exp_results[0].eval_results))]
        for exp_result in exp_results:
            for epoch_result in exp_result.epoch_results:
                grouped_epoch_results[epoch_result.epoch].append(epoch_result)
            for j, eval_result in enumerate(exp_result.eval_results):
                grouped_conf_results[j].append(eval_result)

        avg_epoch_results = []
        for (epoch, epoch_results) in grouped_epoch_results.items():
            avg_epoch_results.append((epoch, combine_epoch_fn(epoch_results)))
        avg_epoch_results = [r for (_, r) in sorted(avg_epoch_results)]
        avg_conf_results = []
        for conf_results in grouped_conf_results:
            avg_conf_results.append(combine_eval_fn(conf_results))
        return ExperimentResult(exp_results[0].config, avg_epoch_results, avg_conf_results)

    def as_dataframe(self):
        data = defaultdict(list)
        for exp_result in self.results:
            config = exp_result.config
            print(config)
            loss = get_loss_abbrev(config)
            task = get_task_abbrev(config)
            for j, eval_result in enumerate(exp_result.eval_results):
                conf_abbrev = get_conf_abbrev(config['confidences'][j])
                method_abbrev = loss + " (" + conf_abbrev + ")"
                data['loss'].append(loss)
                data['method'].append(method_abbrev)
                data['task'].append(task)
                for metric_name in eval_result.as_dict():
                    if metric_name not in ['f1', 'matthews_correlation']:
                        data[metric_name].append(eval_result[metric_name])
        return pd.DataFrame(data=dict(data))


def get_task_abbrev(config):
    task = config['task']['name'] if 'task' in config else "-"
    if task == 'glue':
        task = config['task']['subtask']
    return task


def get_conf_abbrev(cconfig):
    if cconfig['name'] == 'ts':
        return 'trustscore'
        # return 'ts({}, {})'.format(cconfig['alpha'], cconfig['max_sample_size'])
    elif cconfig['name'] == 'mcd':
        if cconfig['aggregator'] == "mean":
            return 'mcdm'
            # return 'mcdm({})'.format(cconfig['n_forward_passes'])
        elif cconfig['aggregator'] == "negvar":
            return 'mcdv'
            # return 'mcdv({})'.format(cconfig['n_forward_passes'])
    elif cconfig['name'] == 'max_non_abstain':
        return 'max_prob'
    else:
        return cconfig['name']

def get_loss_abbrev(config):
    if 'regularizer' not in config:
        return 'basic'
    else:
        lconfig = config['regularizer']
        if lconfig['name'] == 'ereg':

            # return 'ereg'
            # return 'ereg({})'.format(lconfig['lambda_param'])
            return 'ereg({})'.format(config['bsz'])
        elif lconfig['name'] == 'dac':
            return 'dac'
            # return 'dac({},{})'.format(lconfig['alpha_final'], lconfig['alpha_init_factor'])
        else:
            return lconfig['name']


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


def plot_training_metric(exp_results, metric_name):
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

def plot_evaluation_metric(result_db, metric_name):
    df = result_db.as_dataframe()
    with open('final_results.csv', 'w') as writer:
        df = df.sort_values(by=['method'])
        writer.write(df.to_csv(index=False))
    sns.set_theme(style="whitegrid")
    sns.violinplot(y="method", x=metric_name, hue="task",
                   data=df, orient="h", inner="stick")
    plt.gcf().subplots_adjust(left=0.35)
    plt.show()


def compete(df, metric_name, method1, baseline):
    df1 = df[df['method']==method1]
    df2 = df[df['method']==baseline]
    df1 = df1[['task', metric_name]]
    df2 = df2[['task', metric_name]]
    results1 = defaultdict(list)
    for (task, value) in df1.values.tolist():
        results1[task].append(value)
    results2 = defaultdict(list)
    for (task, value) in df2.values.tolist():
        results2[task].append(value)
    results1 = dict(results1)
    results2 = dict(results2)
    differences = []
    for key in results1:
        values1, values2 = results1[key], results2[key]
        shuffle(values1)
        shuffle(values2)
        min_length = min(len(values1), len(values2))
        values1 = values1[:min_length]
        values2 = values2[:min_length]
        for v1, v2 in zip(values1, values2):
            # differences.append(v1 - v2)
            differences.append(int(v1>v2))
    compete_df = pd.DataFrame(data={'versus {}'.format(baseline): [method1] * len(differences),
                                    'likelihood of improved {}'.format(metric_name): differences})
    return compete_df

def create_versus_df(result_df, metric_name, baseline, methods):
    sub_dfs = []
    for method in methods:
        sub_dfs.append(compete(result_df, metric_name, method, baseline))
    return pd.concat(sub_dfs)


def viz_versus(result_db, metric_name):
    baseline = 'basic (max_prob)'
    methods = ['basic (mcdv)',
               'basic (mcdm)',
               'ereg (max_prob)',
               'dac (max_prob)',
               'basic (trustscore)',
               'basic (random)']
    result_df = result_db.as_dataframe()
    versus_df = create_versus_df(result_df, metric_name, baseline, methods)
    sns.set_theme(style="whitegrid")
    sns.pointplot(y="versus {}".format(baseline),
                  x='likelihood of improved {}'.format(metric_name),
                  data=versus_df, orient="h", join=False,
                  order=methods)
    plt.gcf().subplots_adjust(left=0.35)
    plt.show()


def example_pr_curve(conf="g1"):
    d = {'precision': [1, 1, 1, 1, 1, 4/5, 4/6, 4/7, 5/8, 5/9, 6/10,
                       1, 1, 1, 1, 1, 1, 1, 6/7, 6/8, 6/9, 6/10,
                       0, 0, 0, 0, 1/5, 2/6, 3/7, 4/8, 5/9, 6/10],
         'recall': [0, 1/6, 2/6, 3/6, 4/6, 4/6, 4/6, 4/6, 5/6, 5/6, 1,
                    0, 1/6, 2/6, 3/6, 4/6, 5/6, 1, 1, 1, 1, 1,
                    0, 0, 0, 0, 1/6, 2/6, 3/6, 4/6, 5/6, 1],
         'confidence': ['g1']*11 + ['g2']*11 + ['g3']*10 }
    df = pd.DataFrame(data=d)
    df1 = df[df["confidence"]==conf]
    g = sns.FacetGrid(df1, hue="confidence", height=8)
    sns.set(font_scale=3)
    g.map(plt.scatter, "recall", "precision")
    g.map(plt.plot, "recall", "precision")
    g.set(ylim=(-0.01, 1.01))
    g.set(xlim=(-0.01, 1.01))


def example_pr_curve2(conf="g1"):
    d = {'precision': [9/10, 8/9, 7/8, 6/7, 5/6, 4/5, 3/4, 2/3, 2/2, 1/1, 1,
                       9/10, 9/9, 8/8, 7/7, 6/6, 5/5, 4/4, 3/3, 2/2, 1/1, 1,
                       9/10, 8/9, 7/8, 6/7, 5/6, 4/5, 3/4, 2/3, 1/2, 0/1],
         'recall': [9/9, 8/9, 7/9, 6/9, 5/9, 4/9, 3/9, 2/9, 2/9, 1/9, 0,
                    9/9, 9/9, 8/9, 7/9, 6/9, 5/9, 4/9, 3/9, 2/9, 1/9, 0,
                    9/9, 8/9, 7/9, 6/9, 5/9, 4/9, 3/9, 2/9, 1/9, 0/9],
         'confidence': ['g1']*11 + ['g2']*11 + ['g3']*10 }
    df = pd.DataFrame(data=d)
    df1 = df[df["confidence"]==conf]
    g = sns.FacetGrid(df1, hue="confidence", height=8)
    g.map(plt.scatter, "recall", "precision")
    g.map(plt.plot, "recall", "precision")
    g.set(ylim=(-0.01, 1.01))
    g.set(xlim=(-0.01, 1.01))


def get_dataframe(directory):
    result_db = ResultDatabase.load(directory)
    return result_db.as_dataframe()


def main(directory, metric_name):
    result_db = ResultDatabase.load(directory)
    plot_evaluation_metric(result_db, metric_name)
    # viz_versus(result_db, metric_name)

if __name__ == '__main__':
    direc = sys.argv[1]
    metric = sys.argv[2]
    main(direc, metric)
