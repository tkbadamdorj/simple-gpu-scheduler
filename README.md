# gpu-hyperparameter-search
Hyperparameter search wrapper that uses multiple GPUs.

Advantages:
- Automates hyperparameter search
- Automatically picks free GPUs
- Works for any train.py file that allows hyperparameter specification using command line arguments (see demo)
- Both grid search and random search available

Assumes:
- train.py can have GPU specified using --gpu_num `GPU_ID` file
- train.py keeps track of metrics in separate files e.g. Tensorboard logs

Please note that models are pre-assigned to GPUs and each model is run sequentially on a given GPU.

Example:
If there are 9 models to try out, and we have three GPUs, each GPU is assigned three models.
All three GPUs will start training the first models they were assigned. Once a GPU finishes training a model, it will start training the next model it was assigned.

Useful when models are of similar size and run for the same number of epochs.
