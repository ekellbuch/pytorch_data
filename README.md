# Pytorch data

A library with pytorch datasets compatible with pytorch lightning

Datasets
----------

|      Dataset       | 
|:------------------:|
|      CIFAR-10      |
|     CIFAR10_C      |
|      CINIC-10      |
|     CIFAR-10.1     |
|     CIFAR-100      |
|  CIFAR-100 Coarse  |
| CIFAR-10-LongTail  |
| CIFAR-100-LongTail |
|      ImageNet      |
|    iNaturalist     |
|       MNIST        |
|        UCI         |


## Installation instructions:

Instructions using conda

Check name of existing conda environments (to not overwrite them): 
```
conda list
```
Create a new environment as follows (can use 3.8+):
```
conda create -n env_name python=3.8
```
Now move into the root directory of this repo:
```
cd /path/to/this/repo
```

Activate your new environment, install dependencies and python package: 
```
conda activate env_name
conda install pip 
pip install -r requirements.txt
pip install -e ./src
```

References
-------------
[CIFAR-10.1](https://github.com/modestyachts/CIFAR-10.1)

[CIFAROOD](https://github.com/cellistigs/cifar10_ood)

