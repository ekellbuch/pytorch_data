import numpy as np
import pytorch_lightning as pl
import requests
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


def count_classes(data, num_classes=10, loader='train'):
  " count the # classes in a dataset"
  loaders = {
    'train': data.train_dataloader,
    'val': data.val_dataloader,
    'test': data.test_dataloader,
  }
  labels = torch.zeros(num_classes, dtype=torch.long)
  for idx, batch in enumerate(loaders[loader]()):
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
    self.hparams.update(args)
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
    # Function called to split train_set into train_dataset and val_dataset
    # valid_set is the same data as train_set wo augmentation
    assert self.valid_size < 1, "valid_size should be less than 1"
    assert self.valid_size >= 0, "valid_size should be greater than or eq to 0"

    if self.valid_size == 0:
        self.train_dataset = train_set
        self.val_dataset = test_set
    else:
        # split train set into train and val set, keeping class proportion.
        targets_np = np.array(train_set.targets, dtype=np.int64)
        classes, class_count = np.unique(targets_np, return_counts=True)
        new_train_indices = []
        new_val_indices = []
        for the_class, the_class_count in zip(classes, class_count):
            idx = np.where(targets_np == the_class)[0]
            if self.seed is not None:
                np.random.seed(self.seed)
                np.random.shuffle(idx)
            split = int(np.floor((1-self.valid_size) * the_class_count))
            train_count = idx[:split]
            val_count = idx[split:]
            new_train_indices.extend(train_count)
            new_val_indices.extend(val_count)

        self.train_indices = new_train_indices
        self.val_indices = new_val_indices

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
