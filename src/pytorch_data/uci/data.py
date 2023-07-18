import numpy as np
from typing import Any, Optional

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

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
