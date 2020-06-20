import os
from tabulate import tabulate
import json

def visualize_results(log_dir):
    # return if no logs exist
    if not os.path.exists(log_dir):
        return None

    models = os.listdir(log_dir)
    model_dirs = [f'{log_dir}/{model}' for model in models]

    hparams = []
    metrics = []

    for model_dir in model_dirs:
        hparam_path = f'{model_dir}/hparams.json'
        metric_path = f'{model_dir}/metrics.json'

        if not os.path.exists(hparam_path) or not os.path.exists(metric_path):
            print('hparams.json or metrics.json does not exist')

        with open(hparam_path, 'r') as f:
            hparam = json.load(f)

        hparams.append(hparam)

        with open(metric_path, 'r') as f:
            metric = json.load(f)

        metrics.append(metric)


    hparam_names = [name for name in hparams[0]]
    metric_names = [name for name in metrics[0]]

    headers = hparam_names + metric_names

    table = [[hparams[i][param] for param in hparam_names] +
             [metrics[i][metric] for metric in metric_names]
             for i in range(len(hparams))]

    print(tabulate(table, headers=headers, tablefmt='github'))