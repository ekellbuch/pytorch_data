"""
Edited from
https://github.com/facebookresearch/classifier-balancing/blob/main/data/dataloader.py
update init
"""
import json
import os

from PIL import Image
from torchvision import transforms as T
from torchvision.datasets import INaturalist
from tqdm import tqdm
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

from ..utils import BaseDataModule, stream_download

dir_path = os.path.dirname(os.path.realpath(__file__))

__all__lp = [
     "iNaturalistData",
     "iNaturalist18Data",
]


class INaturalist18(INaturalist):

  def __getitem__(self, index: int) -> Tuple[Any, Any]:
    """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where the type of target specified by target_type.
    """

    cat_id, fname = self.index[index]
    img = Image.open(os.path.join(self.root, self.all_categories[cat_id], fname))

    target: Any = []
    for t in self.target_type:
      if t == "full":
        target.append(cat_id)
      else:
        target.append(self.categories_map[cat_id][t])
    target = tuple(target) if len(target) > 1 else target[0]

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target



class iNaturalistData(BaseDataModule):
  def __init__(self, args):
    super().__init__(args)
    self.mean = (0.466, 0.471, 0.380)
    self.std = (0.195, 0.194, 0.192)

    self.pin_memory = False

  def prepare_data(self):
      # download
      root_dir = os.path.join(self.hparams.data_dir, "INaturalist21")
      INaturalist(root_dir, version="2021_train", download=True)
      INaturalist(root_dir, version="2021_valid", download=True)

  def setup(self, stage=None):
      # Assign train/val dataset for use in dataloaders
      root_dir = os.path.join(self.hparams.data_dir, "INaturalist21")
      train_set = INaturalist(root_dir, "2021_train", transform=self.train_transform())
      valid_set = INaturalist(root_dir, "2021_train", transform=self.valid_transform())
      test_set = INaturalist(root_dir, "2021_valid", transform=self.valid_transform())

      self._split_train_set(train_set, test_set, valid_set)

  def train_transform(self, aug=True):
    if aug is True:
      transform = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(self.mean, self.std),
      ])
      return transform
    else:
      return self.valid_transform()

  def valid_transform(self):
    transform = T.Compose([
      T.Resize(256),
      T.CenterCrop(224),
      T.ToTensor(),
      T.Normalize(self.mean, self.std),
    ])
    return transform


class iNaturalist18Data(iNaturalistData):
  def __init__(self, args):
    super().__init__(args)

  def prepare_data(self):
      # download
      root_dir = os.path.join(self.hparams.data_dir, "INaturalist18")
      INaturalist18(root_dir, version="2018", download=False)

  def setup(self, stage=None):
      # Assign train/val dataset for use in dataloaders
      root_dir = os.path.join(self.hparams.data_dir, "INaturalist18")
      train_set = INaturalist18(root_dir, "2018", transform=self.train_transform(), train=True)
      valid_set = INaturalist18(root_dir, "2018", transform=self.valid_transform(), train=True)
      test_set = INaturalist18(root_dir, "2018", transform=self.valid_transform(), train=False)

      self._split_train_set(train_set, test_set, valid_set)
