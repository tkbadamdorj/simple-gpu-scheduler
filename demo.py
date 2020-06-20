import argparse

from vis_utils import visualize_results
from hp_search import HPSearch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid_search', action='store_true',
                        help='set this flag to do grid search')
    parser.add_argument('--num_random', type=int, default=3,
                        help='number of random hyperparameters to pick if not doing grid search')
    parser.add_argument('--train_file_path', type=str, default='demo_train.py',
                        help='main training file that will be run using command like python train.py')
    parser.add_argument('--virtual_env_dir', type=str, default='.env',
                        help='directory containing virtual environment e.g. if .env is directory, environment'
                             'will be activated using . .env/bin/activate')
    parser.add_argument('--leave_num_gpus', type=int, default=0,
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

    all_flags = {

    }

    hp_search = HPSearch(all_hparams, all_flags, args)

    hp_search.search()

    # try to tabulate results if hparams.json and metrics.json files exist in each log directory
    visualize_results(args.log_dir)

    print('hyperparameter search finished.')