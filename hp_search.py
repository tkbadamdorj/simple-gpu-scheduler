"""
Author: Taivanbat "TK" Badamdorj
github.com/taivanbat
"""

import itertools
import subprocess
from multiprocessing import Process
from pprint import pprint
from random import choice
from subprocess import DEVNULL
import os

from gpu_utils import GPUMemoryUtils
from param_utils import *


class HPSearch:
    def __init__(self, all_hparams, all_flags, args):
        """
        Speeds up hyperparameter search by parallelizing over GPUs

        Argument:
            all_hparams: dict
                keys are hyperparameter names
                values are list of values to try out for each
                hyperparameter
            all_flags: dict
                key is name of flag
                value is "fixed" or "param"
                    if "fixed", this flag is used for all experiments
                    if "param", we experiment with this flag being
                    present or absent when training models i.e.
                    it's another parameter
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
        # preprocess parameters
        all_hparams = process_hparams(all_hparams)
        all_flags = process_flags(all_flags)
        self.all_params = all_hparams + all_flags

        print('OPTIMIZING OVER:')
        pprint(self.all_params)

        # search settings
        self.all_hparams = all_hparams
        self.all_flags = all_flags
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
        params = self._get_params()

        # assign models to GPUs
        gpus, params_gpu = self._gpu_scheduler(params)

        # no available GPUs
        if gpus is None:
            return None

        # train on GPUs
        self._train_on_gpus(gpus, params_gpu)


    """PROCESS HYPERPARAMETERS"""
    def _get_params(self):
        """
        Get list of hyperparameters to try out

        Return:
            params: list
                contains hyperparameters for each model as
                dictionary
        """
        param_grid = self._get_param_grid(self.all_params) if self.grid_search else {}

        # set number of sessions to length of number of combinations
        # for grid search and number of random choices for random search
        num_models = len(param_grid) if self.grid_search else self.num_random

        params = []

        for model_num in range(num_models):
            chosen_params = []

            if self.grid_search:
                # pick the next combination in the grid
                for i, param in enumerate(self.all_params):
                    chosen_params.append(param_grid[model_num][i])
            else:
                for param in self.all_params:
                    chosen_params.append(choice(param.values))

            params.append(chosen_params)

        return params


    def _get_param_grid(self, all_params):
        """
        Find all combinations of hyperparameters for grid search

        Argument:
            all_params: dict
                keys are hyperparameter names
                values are list of values to try out for each hyperparameter

        Return:
            list containing all possible combinations of hyperparameters
        """
        all_values = [param.values for param in all_params]
        param_grid = list(itertools.product(*all_values))

        return param_grid


    """SCHEDULE GPU JOBS"""
    def _gpu_scheduler(self, params):
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
        print(f'Running {len(params)} processes. Leaving {self.leave_num_gpus} GPU(s) free.')

        gpus = self._pick_gpus(self.leave_num_gpus)

        if gpus is None:
            return None, None

        # number of GPUs we're using
        num_use_gpus = len(gpus)
        params_gpu = self._models_to_gpus(params, num_use_gpus)

        return gpus, params_gpu


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
            chosen_gpus = self.gpu_memory.get_free_gpus(self.memory_threshold)
        # pick GPUs but leave leave_num_gpus free
        else:
            free_gpus = self.gpu_memory.get_free_gpus(self.memory_threshold)
            max_idx = len(free_gpus) - leave_num_gpus
            chosen_gpus = free_gpus[:max_idx]

        return chosen_gpus


    def _models_to_gpus(self, params, num_use_gpus):
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
        params_gpu = [[] for i in range(num_use_gpus)]
        for i in range(len(params)):
            params_gpu[i % num_use_gpus].append(params[i])

        return params_gpu


    """TRAIN"""
    def _train_on_gpus(self, gpus, params_gpu):
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
            p = Process(target=self._run_trainer_single_gpu, args=(params_gpu[i], gpu))
            p.start()
            proc.append(p)

        for p in proc:
            p.join()


    def _run_trainer_single_gpu(self, params, gpu):
        """
        Run models defined by hparams on a single GPU sequentially

        Argument:
            params: list
                contains param of each model as list
            gpu:
                id of GPU to use
        """
        for i, param in enumerate(params):
            print('----------------------------------')
            print(f'Running process {i} on gpu {gpu}')
            pprint(param)
            print('----------------------------------\n')
            p = self._run_trainer(param, gpu)
            p.wait()
            print(f'process {i} finished on gpu {gpu}')


    def _run_trainer(self, param, gpu_num):
        """
        Builds up command to run in command line and runs it

        Argument:
            param: list
                params of model
            gpu_num: int
                specifies which gpu to use

        Return:
            p: running subprocess
        """
        param_cmd = ' '.join([parameter.get_command() for parameter in param])

        # set virtual environment (if applicable)
        activate_venv = f'. {self.virtual_env_dir}/bin/activate &&' \
            if self.virtual_env_dir is not None else ''

        gpu_cmd = f'export CUDA_VISIBLE_DEVICES={gpu_num} &&'

        if not os.path.exists(self.train_file_path):
            print(f'{self.train_file_path} does not exist.')
            return None

        # build up final command
        cmd = f'{activate_venv} {gpu_cmd} python {self.train_file_path} {param_cmd}'

        # remove unnecessary whitespace
        cmd = ' '.join(cmd.split())

        # set to DEVNULL to suppress output from training script
        p = subprocess.Popen(cmd, shell=True, stdout=DEVNULL, stderr=DEVNULL)

        return p
