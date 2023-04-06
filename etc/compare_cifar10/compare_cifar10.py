"""
compare cifar 10 balanced and imabalced datasets
"""

from pytorch_data.cifar.data import CIFAR10Data
from pytorch_data.cifar.data_imbalanced import IMBALANCECIFAR10Data
from pytorch_data.cifar.data_influenced import CIFAR100HighInfData
from tqdm import tqdm
import numpy as np
import torch

from ml_collections import config_dict
cfg = config_dict.ConfigDict()

cfg.data_dir= "/home/ekellbuch/pytorch_datasets"
cfg.batch_size= 128
cfg.num_workers= 8
cfg.train_shuffle = False
args = cfg

ind_data = CIFAR10Data(args)
imb_data = IMBALANCECIFAR10Data(args)


#%%
def count_classes(data, num_classes=10, loader='val'):
  loaders = {
    'train' : data.train_dataloader(),
     'val': data.val_dataloader(),
  }
  labels = torch.zeros(num_classes, dtype=torch.long)
  #for idx, batch in tqdm(enumerate(data.train_dataloader(shuffle=False, aug=False))):
  for idx, batch in tqdm(enumerate(loaders[loader])):#shuffle=False, aug=False))):
      labels += torch.bincount(batch[1], minlength=num_classes)
  #print(labels)
  return labels

count_classes(ind_data)
# tensor([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000])
count_classes(imb_data)
# tensor([5000, 2997, 1796, 1077,  645,  387,  232,  139,   83,   50])

#%%
ind_data = CIFAR100HighInfData(args)
count_classes(ind_data, num_classes=100, loader='val').sum()
#%%