# gpu-hparam-search-scheduler
(or GHSS for short I guess)

This wrapper allows you to train different models on multiple GPUs in parallel. You don't have to add anything to code that already works. If you can specify hyperparameters for your `train.py` file by setting different flags, then you can use this. 

## How it works 

You just need to specify the hyperparameters and flags that you would like to search over in two dictionaries, `all_params` and `all_flags` like in `demo.py`. If you want to fix a flag, write `fixed` as the value, and if you want to test a model with and without that flag, write `param` as the value. 

```
all_hparams = {
    'learning_rate': [0.1, 0.01, 0.001],
    'momentum': [0.9, 0.99]
}

all_flags = {
    'normalize': 'fixed',
    'dropout': 'param'
}
```

Then initialize an HPSearch object, and run the search method. The `args` that you need are also defined in `demo.py`

```
hp_search = HPSearch(all_hparams, all_flags, args)

hp_search.search()
```

## Scheduling

Models are pre-assigned to GPUs and models on each GPU are run sequentially. If there are 9 models to try out, and we have three GPUs, each GPU is assigned three models. All three GPUs will start training the first models they were assigned. Once a GPU finishes training a model, it will start training the next model it was assigned.

## Advantages

- Compatible with all machine learning libraries. 
- No need to add anything to your code
- Automates hyperparameter search
- Automatically picks free GPUs with option to leave some free 
- Works for any train.py file that allows hyperparameter specification using command line arguments (see demo)
- Both grid search and random search available

## Disadvantages

- If model training times are very different simple scheduling is not efficient e.g. if GPU 0 is assigned three models that take a long time to train, while GPU 1 is assigned three models that are smaller, GPU 1 may finish very quickly, while GPU 0 marches on.

## Install

Download the package.

```
git clone https://github.com/taivanbat/gpu-hparam-search-scheduler.git 
```

Install a virtual environment and activate it 

```
cd gpu-hyperparameter-search
virtualenv .env && source .env/bin/activate
``` 

Then install all the required packages 

```
pip install -r requirements.txt
```

## Demo

Run the demo 
```
python demo.py
```

```
--grid_search: set this flag to do grid search
--num_random: number of random sets of hyperparameters to pick if not doing grid search
--train_file_path: path to main training file 
--virtual_env_dir: can set directory of a virtual environment that will activate before the train.py is called 
--leave_num_gpus: number of GPUs to leave free. Useful if workstation is shared 
--memory_threshold: GPU that uses less than memory_threshold in MB is considered `not in use`
--pick_last_free_gpu: if flag is set, uses last free GPU if there is only one GPU free
--log_dir: where to store model logs 
```

### The output will look something like this: 

```
OPTIMIZING OVER:
[learning_rate: [0.1, 0.01, 0.001], momentum: [0.9, 0.99]]
Running 3 processes. Leaving 0 GPU(s) free.
----------------------------------
Running process 0 on gpu 1
[learning_rate: 0.1, momentum: 0.99]
----------------------------------

----------------------------------
Running process 0 on gpu 3
[learning_rate: 0.001, momentum: 0.99]
----------------------------------

process 0 finished on gpu 1
----------------------------------
Running process 1 on gpu 1
[learning_rate: 0.001, momentum: 0.99]
----------------------------------

|   learning_rate |   momentum | log_dir   | cuda   |   accuracy |
|-----------------|------------|-----------|--------|------------|
|           0.001 |       0.99 | logs      | True   |     0.2595 |
|           0.1   |       0.99 | logs      | True   |     0.1    |
|           0.001 |       0.99 | logs      | True   |     0.2616 |
```
