import torch
from tqdm import tqdm
import numpy as np
import requests
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset

def count_classes(data, num_classes=10, loader='train'):
  loaders = {
    'train': data.train_dataloader,
    'val': data.val_dataloader,
  }
  labels = torch.zeros(num_classes, dtype=torch.long)
  for idx, batch in tqdm(enumerate(loaders[loader]())):
    labels += torch.bincount(batch[1], minlength=num_classes)
  return labels


def split_dataset(dataset, split=0.9):
  shuffled_indices = np.random.permutation(len(dataset))
  train_idx = shuffled_indices[:int(split* len(dataset))]
  val_idx = shuffled_indices[int(split * len(dataset)):]
  return train_idx, val_idx


def stream_download(dataurl, download_path):
  """helper function to monitor downloads.
  :param dataurl: path where data is located.
  :param download_path: local path (include filename) where we should write data.

  """
  r = requests.get(dataurl, stream=True)

  # Total size in Mebibyte
  total_size = int(r.headers.get("content-length", 0))
  block_size = 2 ** 20  # Mebibyte
  t = tqdm(total=total_size, unit="MiB", unit_scale=True)

  with open(download_path, "wb") as f:
    for data in r.iter_content(block_size):
      t.update(len(data))
      f.write(data)
  t.close()


  if total_size != 0 and t.n != total_size:
    raise Exception("Error, something went wrong")


class BaseDataModule(pl.LightningDataModule):
  def __init__(self, args):
    super().__init__()
    self.hparams = args
    self.valid_size = args.get("valid_size", 0)

    self.seed = args.get("seed", None)
    self._make_generator_from_seed()

    # parameters fed to dataloader
    self.drop_last = False
    self.pin_memory = True

    assert self.hparams.batch_size is not None, "batch_size is required"
    assert self.hparams.num_workers is not None, "num_workers is required"
  def _make_generator_from_seed(self):
      if self.seed == None:
          self.generator_from_seed = None
      else:
          self.generator_from_seed = torch.Generator().manual_seed(self.seed)

  def _split_train_set(self, train_set, test_set, valid_set=None):

    # split train and valid set
    assert self.valid_size < 1, "valid_size should be less than 1"
    assert self.valid_size >= 0, "valid_size should be greater than or eq to 0"

    if self.valid_size == 0:
      self.train_dataset = train_set
      self.val_dataset = test_set
    else:
      num_train = len(train_set)
      indices = torch.randperm(num_train,
                               generator=self.generator_from_seed,
                               )
      split = int(np.floor(self.valid_size * num_train))

      self.train_indices = indices[:num_train - split]
      self.val_indices = indices[num_train - split:]

      self.train_dataset = Subset(train_set, self.train_indices)
      self.val_dataset = Subset(valid_set, self.val_indices)

    self.test_dataset = test_set

  def train_dataloader(self, shuffle=True, aug=True):
      if not aug:
          dataset = self.train_noaug_dataset()
      else:
          dataset = self.train_dataset

      dataloader = DataLoader(
          dataset,
          batch_size=self.hparams.batch_size,
          num_workers=self.hparams.num_workers,
          shuffle=shuffle,
          drop_last=self.drop_last,
          pin_memory=self.pin_memory,
          generator=self.generator_from_seed,
      )
      return dataloader

  def val_dataloader(self):

      dataloader = DataLoader(
          self.val_dataset,
          batch_size=self.hparams.batch_size,
          num_workers=self.hparams.num_workers,
          drop_last=self.drop_last,
          pin_memory=self.pin_memory,
      )
      return dataloader

  def test_dataloader(self):

      dataloader = DataLoader(
          self.test_dataset,
          batch_size=self.hparams.batch_size,
          num_workers=self.hparams.num_workers,
          drop_last=self.drop_last,
          pin_memory=self.pin_memory,
      )
      return dataloader

  def train_noaug_dataset(self):
      raise NotImplementedError("Loading train set wo augmentation is NA")