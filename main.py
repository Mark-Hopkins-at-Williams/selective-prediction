import json
import os
import subprocess
import sys
from spred.experiment import Experiment
from spred.analytics import ResultDatabase, show_training_dashboard

MODELS_BASE_DIR = os.getenv('SPRED_MODELS').strip()
if not os.path.isdir(MODELS_BASE_DIR):
    os.mkdir(MODELS_BASE_DIR)
MODELS_DIR = os.path.join(MODELS_BASE_DIR, 'selective-prediction')
if not os.path.isdir(MODELS_DIR):
    os.mkdir(MODELS_DIR)

def simple_command_line(arguments):
    task_config_id = arguments[1]
    train_config_id = arguments[2]
    task_config_file = ".".join([task_config_id, "config", "json"])
    train_config_file = ".".join([train_config_id, "config", "json"])
    task_config_path = os.path.join("config", "task", task_config_file)
    train_config_path = os.path.join("config", "training", train_config_file)
    results_file = ".".join([task_config_id, train_config_id, "results", "json"])
    output_path = os.path.join(MODELS_DIR, results_file)
    exp = Experiment.from_json(task_config_path, train_config_path)
    result_db = ResultDatabase([exp.run()])
    result_db.save(output_path)
    reloaded = ResultDatabase.load(output_path)
    show_training_dashboard(reloaded.results[0])

def config_command_line(config_file, results_dir):
    with open(config_file, 'r') as f:
        config = json.load(f)
    exp_index = 0
    while os.path.exists(results_dir + "." + str(exp_index)):
        exp_index += 1
    results_dir = results_dir + "." + str(exp_index)
    os.mkdir(results_dir)
    for exp in config:
        task_config_id = exp['task']
        train_config_id = exp['training']
        n_trials = exp['n_trials']
        task_config_file = ".".join([task_config_id, "config", "json"])
        train_config_file = ".".join([train_config_id, "config", "json"])
        task_config_path = os.path.join("config", "task", task_config_file)
        train_config_path = os.path.join("config", "training", train_config_file)
        exp = Experiment.from_json(task_config_path, train_config_path)
        result_db = ResultDatabase([exp.run() for _ in range(n_trials)])
        results_file = ".".join([task_config_id, train_config_id, "results", "json"])
        output_path = os.path.join(results_dir, results_file)
        result_db.save(output_path)


if __name__ == "__main__":
    if sys.argv[1] == "-c":
        config_command_line(sys.argv[2], sys.argv[3])
    else:
        simple_command_line(sys.argv)