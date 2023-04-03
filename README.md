# Pytorch data

Support multiple pytorch datasets in pytorch and pytorch lightning

## Installation instructions:

Instructions using conda

Check name of existing conda environments (to not overwrite them): 
```
conda list
```

Create a new environment as follows:: 

```
conda create -n env_name python=3.7
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
