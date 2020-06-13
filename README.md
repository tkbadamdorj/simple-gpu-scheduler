# gpu-hyperparameter-search
Imagine the scenario: you inherited someone else's code and you have to adapt it for a different task. Instead of manually trying out different sets of hyperparameters, you let the computer do the work for you.

This simple wrapper works with any code and is very useful in the beginning and exploratory stages of a project. This project assumes that hyperparameters for the python script that trains and evaluates the model can be passed in as command line arguments. You can specify the hyperparameters that you want to experiment by defining it inside the `hp_search.py` model.

Then you just hit run, and it will take care of everything else. 

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

Example output when running `hp_search.py`: 

```
OPTIMIZING OVER:
{'learning_rate': [0.1, 0.01, 0.001], 'momentum': [0.9, 0.99]}
Running 6 processes. Leaving 1 GPU(s) free.
Running process 0 on gpu 0
{'learning_rate': 0.1, 'momentum': 0.9}
Running process 0 on gpu 1
{'learning_rate': 0.1, 'momentum': 0.99}
Running process 0 on gpu 2
{'learning_rate': 0.01, 'momentum': 0.9}
process 0 finished on gpu 1
Running process 1 on gpu 1
{'learning_rate': 0.001, 'momentum': 0.9}
process 0 finished on gpu 0
Running process 1 on gpu 0
{'learning_rate': 0.01, 'momentum': 0.99}
process 0 finished on gpu 2
Running process 1 on gpu 2
{'learning_rate': 0.001, 'momentum': 0.99}
process 1 finished on gpu 1
process 1 finished on gpu 0
process 1 finished on gpu 2
|   learning_rate |   momentum |   gpu_num | log_dir   | cuda   |   accuracy |
|-----------------|------------|-----------|-----------|--------|------------|
|           0.1   |       0.9  |         0 | logs      | True   |     0.1    |
|           0.1   |       0.99 |         1 | logs      | True   |     0.1    |
|           0.01  |       0.99 |         0 | logs      | True   |     0.1    |
|           0.001 |       0.99 |         2 | logs      | True   |     0.2583 |
|           0.001 |       0.9  |         1 | logs      | True   |     0.5526 |
|           0.01  |       0.9  |         2 | logs      | True   |     0.2806 |
hyperparameter search finished.
```
