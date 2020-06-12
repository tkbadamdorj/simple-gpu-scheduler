"""
Author: Taivanbat "TK" Badamdorj
github.com/taivanbat

Hyperparameter search wrapper that uses multiple GPUs.

Advantages:
    - Automates hyperparameter search
    - Automatically picks free GPUs
    - Works for any train.py file that allows hyperparameter specification using
    command line arguments (see demo)
    - Both grid search and random search available

Assumes:
    - train.py can have GPU specified using --gpu_num `GPU_ID` file
    - train.py keeps track of metrics in separate files e.g. Tensorboard logs

Please note that models are pre-assigned to GPUs and each model is run sequentially
on a given GPU.

Example:
    If there are 9 models to try out, and we have three GPUs,
    each GPU is assigned three models.

    All three GPUs will start training the first models they were assigned.
    Once a GPU finishes training a model, it will start training the
    next model it was assigned.

Useful when models are of similar size and run for the same number of epochs.
"""

import argparse
import os
import json

import numpy as np

import subprocess
from subprocess import DEVNULL
from multiprocessing import Process
from pprint import pprint

from random import choice
import itertools

from tabulate import tabulate

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
            return None

        with open(hparam_path, 'r') as f:
            hparam = json.load(f)

        hparams.append(hparam)

        with open(metric_path, 'r') as f:
            metric = json.load(f)

        metrics.append(metric)


    hparam_names = [name for name in hparams[0]]
    metric_names =  [name for name in metrics[0]]

    headers = hparam_names + metric_names

    table = [[hparams[i][param] for param in hparam_names] +
             [metrics[i][metric] for metric in metric_names]
             for i in range(len(hparams))]

    print(tabulate(table, headers=headers, tablefmt='github'))

class GPUMemoryUtils:
    def get_used_gpu_memory(self):
        """
        Adapted code from mjstevens777
        https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/3

        Get the current GPU usage.

        Return:
            gpu_memory: numpy array
                memory usage as integers in MB.
        """
        result = subprocess.check_output(
            [
                'nvidia-smi', '--query-gpu=memory.used',
                '--format=csv,nounits,noheader'
            ], encoding='utf-8')

        # Convert lines into list
        gpu_memory = [int(x) for x in result.strip().split('\n')]
        gpu_memory = np.array(gpu_memory)

        return gpu_memory


    def get_free_gpus(self, memory_threshold=150):
        """
        Get indices of free GPUs.

        A GPU is free if its used memory is less than memory_threshold

        Argument:
            memory_threshold: int
        Return:
            free_gpus: numpy array
                indices of free GPUs
        """
        used_gpu_memory = self.get_used_gpu_memory()
        free_gpus = np.flatnonzero(used_gpu_memory < memory_threshold)

        return free_gpus


    def get_num_free_gpus(self, memory_threshold=150):
        """
        Get number of available GPUs

        Argument:
            memory_threshold: int
        Return:
            num_free_gpus: int
                number of free GPUs
        """
        used_gpu_memory = self.get_used_gpu_memory()
        num_free_gpus = np.sum(used_gpu_memory < memory_threshold)

        return num_free_gpus


class HPSearch:
    def __init__(self, all_hparams, args):
        """
        Speeds up hyperparameter search by parallelizing over GPUs

        Argument:
            all_hparams: dict
                keys are hyperparameter names
                values are list of values to try out for each
                hyperparameter
            args:
                grid_search: bool
                    set True to enable grid search
                num_random: int
                    number of random hyperparameters to pick if
                    doing random search

                train_file_path: str
                    path to training file e.g. train.py or
                    trainer/train.py
                virtual_env_dir: str
                    path to activate virtual environment. If empty,
                    virtual environment is not activated before training
                leave_num_gpus: int
                    number of GPUs to leave free (useful in case
                    workstation is shared)
                memory_threshold: int
                    GPU that uses less memory than this threshold is
                    considered free
                pick_last_free_gpu: bool
                    picks last free GPU if no other free GPU
        """
        # search settings
        self.all_hparams = all_hparams
        self.grid_search = args.grid_search
        self.num_random = args.num_random

        # training settings
        self.train_file_path = args.train_file_path
        self.virtual_env_dir = args.virtual_env_dir
        self.leave_num_gpus = args.leave_num_gpus
        self.memory_threshold = args.memory_threshold

        # initialize GPU memory utils
        self.gpu_memory = GPUMemoryUtils()
        self.pick_last_free_gpu = args.pick_last_free_gpu


    def search(self):
        """
        Main function that:
            1) gets hyperparameters for each model
            2) assigns models to separate GPUs
            3) runs assigned models on each GPU separately
        """

        # get model hyperparameters
        hparams = self._get_hparams()

        # assign models to GPUs
        gpus, hparams_gpu = self._gpu_scheduler(hparams)

        # no available GPUs
        if gpus is None:
            return None

        # train on GPUs
        self._train_on_gpus(gpus, hparams_gpu)

    """PROCESS HYPERPARAMETERS"""
    def _get_hparams(self):
        """
        Get list of hyperparameters to try out

        Return:
            hparams: list
                contains hyperparameters for each model as
                dictionary
        """
        hp_grid = self._get_hparam_grid(self.all_hparams) if self.grid_search else {}

        # set number of sessions to length of number of combinations
        # for grid search and number of random choices for random search
        num_models = len(hp_grid) if self.grid_search else self.num_random

        hparams = []

        for model_num in range(num_models):
            chosen_hp = {}

            if self.grid_search:
                # pick the next combination in the grid
                for i, param in enumerate(self.all_hparams):
                    chosen_hp[param] = hp_grid[model_num][i]
            else:
                for param in self.all_hparams:
                    chosen_hp[param] = choice(self.all_hparams[param])

            hparams.append(chosen_hp)

        return hparams


    def _get_hparam_grid(self, all_hparams):
        """
        Find all combinations of hyperparameters for grid search

        Argument:
            all_hparams: dict
                keys are hyperparameter names
                values are list of values to try out for each hyperparameter

        Return:
            list containing all possible combinations of hyperparameters
        """
        all_values = [all_hparams[param] for param in all_hparams]
        hp_grid = list(itertools.product(*all_values))

        return hp_grid


    """SCHEDULE GPU JOBS"""
    def _gpu_scheduler(self, hparams):
        """
        Picks GPUs and schedules models to train on each GPU

        Argument:
            hparams: list
                contains hyperparameters for each model as
                dictionary
            leave_num_gpus: int
                number of GPUs to leave free (useful in case
                workstation is shared)
        """
        print(f'Running {len(hparams)} processes. Leaving {self.leave_num_gpus} GPU(s) free.')

        gpus = self._pick_gpus(self.leave_num_gpus)

        if gpus is None:
            return None, None

        # number of GPUs we're using
        num_use_gpus = len(gpus)
        hparams_gpu = self._models_to_gpus(hparams, num_use_gpus)

        return gpus, hparams_gpu


    def _pick_gpus(self, leave_num_gpus):
        """
        Pick the GPUs to use while leaving leave_num_gpus free

        Returns indices of GPUs to use

        If there is only one free GPU, use that regardless of value of
        leave_num_gpus if pick_last_free_gpu is True

        Argument:
            leave_num_gpus: int
                number of GPUs to leave free (useful in case workstation
                is shared)

        Return:
            gpus: list
                list of GPUs to use
        """
        num_free_gpus = self.gpu_memory.get_num_free_gpus(self.memory_threshold)

        # pick GPUs to use
        if num_free_gpus == 0:
            print('No free GPUs.')
            return None
        # if only one GPU left, use that
        elif num_free_gpus == 1 and self.pick_last_free_gpu:
            gpus = self.gpu_memory.get_free_gpus(self.memory_threshold)
        # pick GPUs but leave leave_num_gpus free
        else:
            gpus = self.gpu_memory.get_free_gpus()[:-leave_num_gpus]

        return gpus


    def _models_to_gpus(self, hparams, num_use_gpus):
        """
        Assigns different models to each GPU

        Example:
            If we use 3 GPUs for 9 different models,
            GPU 0 gets: hparams[0], hparams[3], hparams[6]
            GPU 1 gets: hparams[1], hparams[4], hparams[7]
            GPU 2 gets: hparams[2], hparams[5], hparams[8]

        Argument:
            hparams: list
                contains hyperparameters of each model as dictionary
            num_use_gpus: int
                number of GPUs we use

        Return:
            hparams_gpu: list of lists
                each list item inside hparams_gpu contains
                all hyperparameters to try for a single GPU
        """
        hparams_gpu = [[] for i in range(num_use_gpus)]
        for i in range(len(hparams)):
            hparams_gpu[i % num_use_gpus].append(hparams[i])

        return hparams_gpu


    """TRAIN"""
    def _train_on_gpus(self, gpus, hparams_gpu):
        """
        Run processes on given GPUs in parallel

        Example:
            gpus[0] runs all models defined by hyperparameters in
            hparams_gpu[0]

        Argument:
            gpus: list
                GPU indices to use
            hparams_gpu: list of lists
                each list item inside hparams_gpu contains
                all hyperparameters to try for a single GPU
        """
        proc = []
        for i, gpu in enumerate(gpus):
            p = Process(target=self._run_trainer_single_gpu, args=(hparams_gpu[i], gpu))
            p.start()
            proc.append(p)

        for p in proc:
            p.join()


    def _run_trainer_single_gpu(self, hparams, gpu):
        """
        Run models defined by hparams on a single GPU sequentially

        Argument:
            hparams: list
                contains hparam of each model as dictionary
            gpu:
                id of GPU to use
        """
        for i, hparam in enumerate(hparams):
            pprint(f'Running process {i} on gpu {gpu}')
            pprint(hparam)
            p = self._run_trainer(hparam, gpu)
            p.wait()
            print(f'process {i} finished on gpu {gpu}')


    def _run_trainer(self, hparam, gpu_num):
        """
        Builds up command to run in command line and runs it

        Argument:
            hparam: dictionary
                hyperparameters of model
            gpu_num: int
                specifies which gpu to use

        Return:
            p: running subprocess
        """
        # hyperparameters set as flags
        hparam_cmd = ' '.join([f'--{param} {val}'
                               for param, val in hparam.items()])

        # set virtual environment (if applicable)
        activate_venv = f'. {self.virtual_env_dir}/bin/activate &&' \
            if self.virtual_env_dir is not None else ''

        # build up final command
        cmd = f"{activate_venv} python {self.train_file_path} " \
              f"--gpu_num {gpu_num} {hparam_cmd}"

        # set to DEVNULL to suppress output from training script
        p = subprocess.Popen(cmd, shell=True, stdout=DEVNULL, stderr=DEVNULL)

        return p


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid_search', action='store_true',
                        help='set this flag to do grid search')
    parser.add_argument('--num_random', type=int, default=10,
                        help='number of random hyperparameters to pick if not doing grid search')
    parser.add_argument('--train_file_path', type=str, default='demo.py',
                        help='main training file that will be run using command like python train.py')
    parser.add_argument('--virtual_env_dir', type=str, default=None,
                        help='directory containing virtual environment e.g. if .env is directory, environment'
                             'will be activated using . .env/bin/activate')
    parser.add_argument('--leave_num_gpus', type=int, default=1,
                        help='number of GPUs to keep free. useful for shared workstations')
    parser.add_argument('--memory_threshold', type=int, default=150,
                        help='GPU that uses less than memory_threshold in MB is considered `not in use`')
    parser.add_argument('--pick_last_free_gpu', action='store_true',
                        help='pick the last free gpu if there is only one gpu remaining')
    parser.add_argument('--log_dir', type=str, default='logs')
    args = parser.parse_args()


    all_hparams = {
        'learning_rate': [0.1, 0.01, 0.001],
        'momentum': [0.9, 0.99]
    }

    print('OPTIMIZING OVER:')
    pprint(all_hparams)

    hp_search = HPSearch(all_hparams, args)

    hp_search.search()

    # try to tabulate results if hparams.json and metrics.json files exist in each log directory
    visualize_results(args.log_dir)

    print('hyperparameter search finished.')