import os
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import Any, Iterable, Optional, Tuple, Union

from .raw_data import BaselineDataset

__all__ = ["UCIDataset",
           ]

__all__lp = [
    "UCIDataModule",
]

class UCIDataset(Dataset):
  def __init__(self,
               data_dir: str,
               dataset: str,
               train=False,
               N_train=1000,
               seed: Optional[int] = None,
               N_test: Optional[int] = None,
               transform: Optional[Any] =None,
               target_transform: Optional[Any] =None,
               **kwargs):

    if seed is not None:
      np.random.seed(seed)

    self.data_dir = data_dir
    self.transform = transform
    self.target_transform = target_transform

    # read data
    data = BaselineDataset(
      data_dir=data_dir,
      dataset=dataset,
      N=N_train,
      N_test=N_test,
      seed=seed,
      **kwargs,
    )

    data.get_dataset()
    X_train, y_train = data.X, data.y
    X_test, y_test = data.X_test, data.y_test

    if train == False:
      self.features = X_test.squeeze(0)
      self.targets = y_test.squeeze(0).unsqueeze(-1)
    else:
      self.features = X_train.squeeze(0)
      self.targets = y_train.squeeze(0).unsqueeze(-1)

  def __len__(self):
    return len(self.features)

  def __getitem__(self, idx):
    feature, target = self.features[idx], self.targets[idx]
    if self.transform:
      feature = self.transform(feature)
    if self.target_transform:
      target = self.target_transform(target)
    return feature, target


class UCIDataModule(pl.LightningDataModule):
  def __init__(self, args):
    super().__init__()
    self.hparams = args
    N_train = self.hparams.get('N_train', 1000)
    N_test = self.hparams.get('N_test', None)
    seed = self.hparams.get('seed', None)

    self.test_data = UCIDataset(data_dir=self.hparams.data_dir,
                                  dataset=self.hparams.dataset,
                                  seed=seed,
                                  N_train=N_train,
                                  N_test=N_test,
                                  train=False)

    self.train_data = UCIDataset(data_dir=self.hparams.data_dir,
                                dataset=self.hparams.dataset,
                                seed=seed,
                                N_train=N_train,
                                N_test=N_test,
                                train=True)

  def train_dataloader(self, shuffle=False, aug=False):
    return DataLoader(self.train_data,
                      batch_size=self.hparams.batch_size,
                      num_workers=self.hparams.num_workers,
                      drop_last=False,
                      pin_memory=True,
                      shuffle=shuffle)

  def val_dataloader(self):
    return DataLoader(self.test_data,
                      batch_size=self.hparams.batch_size,
                      num_workers=self.hparams.num_workers,
                      drop_last=False,
                      pin_memory=True, )

  def test_dataloader(self):
    return self.val_dataloader()


class WineDataset(Dataset):
  def __init__(self, root_dir, transform=None,
               target_transform=None,
               seed=0, test_size=1000,
               color=None,
               train=False):
    if seed is not None:
      np.random.seed(seed)
    # super().__init__(root_dir,transform = transform,target_transform = target_transform)
    self.root_dir = root_dir
    self.raw_data_red = pd.read_csv(os.path.join(root_dir, "winequality-red.csv"), sep=";").values.astype(np.float32)
    self.raw_data_white = pd.read_csv(os.path.join(root_dir, "winequality-white.csv"), sep=";").values.astype(
      np.float32)
    self.transform = transform
    self.target_transform = target_transform
    if color == None:
      self.raw_data = np.concatenate([self.raw_data_red, self.raw_data_white], axis=0)
    elif color == "red":
      self.raw_data = self.raw_data_red
    elif color == "white":
      self.raw_data = self.raw_data_white

    # normalize data.
    mean, std = np.mean(self.raw_data, axis=0), np.std(self.raw_data, axis=0)
    self.normed_data = (self.raw_data - mean) / std
    indices = np.random.permutation(range(len(self.raw_data)))
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    if train == False:
      self.features = self.normed_data[test_indices, :-1]
      self.targets = self.normed_data[test_indices, -1:]
    else:
      self.features = self.normed_data[train_indices, :-1]
      self.targets = self.normed_data[train_indices, -1:]

  def __len__(self):
    return len(self.features)

  def __getitem__(self, idx):
    feature, target = self.features[idx], self.targets[idx]
    if self.transform:
      feature = self.transform(feature)
    if self.target_transform:
      target = self.target_transform(target)
    return feature, target


class WineDataModule(pl.LightningDataModule):
  def __init__(self, args):
    super().__init__()
    self.hparams = args
    self.wine_predict = WineDataset(self.hparams.data_dir,
                                    train=False,
                                    color=self.hparams.winecolor,
                                    test_size=1000)
    self.wine_train = WineDataset(self.hparams.data_dir,
                                  train=True,
                                  color=self.hparams.winecolor,
                                  test_size=self.hparams.testset_size)

  def train_dataloader(self, shuffle=False, aug=False):
    return DataLoader(self.wine_train,
                      batch_size=self.hparams.batch_size,
                      num_workers=self.hparams.num_workers,
                      drop_last=False,
                      pin_memory=True,
                      shuffle=shuffle)

  def val_dataloader(self):
    return DataLoader(self.wine_predict,
                      batch_size=self.hparams.batch_size,
                      num_workers=self.hparams.num_workers,
                      drop_last=False,
                      pin_memory=True, )

  def test_dataloader(self):
    return self.val_dataloader()