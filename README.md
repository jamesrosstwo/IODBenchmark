# IODBench

An experiment structure providing easy prototyping and benchmarking of incremental object detection methods.

## Installation

`conda env create -f environment.yaml`

## Features

This repository implements a variety of ways in which you can modify and create your own experiments. This section will discuss each of these extensible modules for use in your own experiments.


### Experiment Structure
The flow of an experiment is as follows:

1. Break up the selected dataset into T splits. Each of these splits are further broken into train and validation splits S and V
2. For each training split, train the strategy on this split.
   1. After training on a given split, evaluate all selected continual metrics on every evaluation split.
   2. Place the values obtained from these continual metrics into a result matrix for each continual metric.
3. Use these result matrices to calculate the final metrics, observing how these continual metrics change over time.

It is also possible to select multiple datasets to run in sequence.

The implementation of this structure can be found in `experiments/experiment.py`. 


### Config Files

Configuration files are used to customize all aspects of the benchmarking pipeline.

The customization allowed by config files involves:
- Output paths
- GPU Usage and device configuration
- Checkpointing
- Datasets
  - Data paths
  - Split organization
  - Preprocessing and transformations
- Strategies
  - Batch size
  - Custom arguments per-strategy such as thresholds, hyperparameters, etc.
- Metrics
  - The set of continual and final metrics used
  - Custom arguments per-metric
- Plugins
  - Set of plugins used 
  - Custom arguments per-plugin
- Loggers used

An example config file demonstrating all of the above by training a naive strategy on the COCO dataset can be found in `confs/local/naive_coco.conf`

### Strategies

Strategies within IODBench are used by the experiment as the endpoint of any inference and training that occurs, as well as the home of any CF-mitigation algorithms.
Strategies tend to rely on `IODModel`s and can often hot-swap between them using configs, although this is not strictly required.

### Models

Used for strategies, `IODModel`s are `torch.nn.Module`s allowing for easy inference, training, prediction, and more.

NOTE: Deformable DETR relies on cython, and has additional setup steps if you intend to use this model
as a baseline.

#### Installation Instructions
Requirements: CUDA>=9.2, GCC>=5.4.
You will also need the CUDA toolkit: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

1. ```sh src/models/ddetr/ops/make.sh```
2. ```python src/models/ddetr/ops/test.py```


### Datasets

Datasets within IODBench extend from `IODDataset`. Datasets themselves are templates for storing metadata, and constructing splits for the dataset.
Datasets will each be associated with both an `IODLoader` and an `IODSplit`. 

`IODSplit`s correspond with torchvision datasets, and implement all of the functionality required by that structure. They keep track of the specific data to be loaded within a split.
Every split within the `IODDataset` will involve two `IODSplit`s, for training and validation.

With these splits constructed, the `IODLoader` is used to load data for the current split into memory. Functionality like data transformations should be kept to within data loaders, and can be modified per-experiment using the config files.


### Metrics

Metrics are divided into two categories: continual and final. Continual metrics are used to determine the performance
of the examined methodologies after every evaluation step. As described in the [experiment structure section](#experiment-structure), continual metrics each build a result matrix R where R^m_{i, j} denotes the result of continual metric m evaluated on split j after training on splits [0, i].

We can then interpret the results of this matrix in many ways. Entries on the main diagonal represent the evaluation of the model on every split as it is trained. Entries below the main diagonal allow us to understand the propensity of the model to forget,
as entries below the main diagonal are evaluation results after we have trained the strategy on new data. In fact, we can observe the change in this metric as we expose the strategy to new, changing data by viewing the columns of this matrix starting from the main diagonal.
We can also interpret the strategy's ability to perform zero-shot learning by observing the results above the main diagonal, as that represents the model's performance on splits that it has not yet been trained on.

Continual metrics are typically estimates of model performance, such as average precision or a loss function of some kind.

Final metrics are used to numerically interpret this result matrix to draw conclusions about the efficacy of this strategy in preventing catastrophic forgetting, or the strategy's ability to perform zero-shot learning. They can also be used to observe the performance of the model as a whole.
Examples of final metrics include continual average precision, backwards weight transfer, and more.

### Plugins

To further extend upon the experiment structure, this repository implements plugins. These plugins are able to inject blocks of code at various points during the experiment process.
Plugins allow for either observations such as VRAM usage, or directly altering the training process by implementing things like replay memory.

Examples of these plugins can be found in `src/plugins`


## Cluster Experiments

By passing an additional "DDP" key to the configuration, it is possible to distribute experiments across
multiple GPUs. 



### Example Run Script
```
#SBATCH --job-name=naive_coco_overlapping
#SBATCH --mem=40G
#SBATCH --qos=normal
#SBATCH --partition='gpu'
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm-%j.out

export PYTHONUNBUFFERED=1
export PYTHONPATH=$HOME/CLBench/venv/bin:/h/jross/CLBench:$PYTHONPATH
export LD_LIBRARY_PATH=/pkgs/cuda-11.3/lib64:/pkgs/cudnn-10.2-v7.6.5.32/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/pkgs/cuda-11.3


source /h/jross/CLBench/venv/bin/activate 
cd /h/jross/CLBench/src
python -m wandb login <key>
python /h/jross/CLBench/src/experiments/__main__.py --conf ../confs/vector/naive_coco.conf
```