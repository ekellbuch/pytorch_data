"""
Edited from
https://github.com/facebookresearch/classifier-balancing/blob/main/data/dataloader.py
"""
import json
import os

from PIL import Image
from torchvision import transforms as T
from torchvision.datasets import INaturalist
from tqdm import tqdm

from ..utils import BaseDataModule, stream_download

dir_path = os.path.dirname(os.path.realpath(__file__))


class iNaturalistData(BaseDataModule):
  def __init__(self, args):
    super().__init__(args)
    self.mean = (0.466, 0.471, 0.380)
    self.std = (0.195, 0.194, 0.192)

    self.pin_memory = False

  def prepare_data(self):
      # download
      root_dir = os.path.join(self.hparams.data_dir, "iNaturalist18")
      INaturalist(root_dir, version="2021_train", download=True)
      INaturalist(root_dir, version="2021_valid", download=True)

  def setup(self, stage=None):
      # Assign train/val dataset for use in dataloaders
      root_dir = os.path.join(self.hparams.data_dir, "iNaturalist18")
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
