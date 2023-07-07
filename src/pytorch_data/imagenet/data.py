import torch
import os
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torchvision.datasets import ImageFolder
from typing import Callable

__all__lp = [
    "ImageNetData",
    "ImageNetDataIndexed",
    "TinyImagenetData"
]


class ImageNetData(pl.LightningDataModule):
  def __init__(self, args):
    super().__init__()
    self.hparams = args
    self.mean = (0.485, 0.456, 0.406)
    self.std = (0.229, 0.224, 0.225)
    self.image_size = 224
    self.train_shuffle = self.hparams.get("train_shuffle", True)

  def train_dataloader(self):
    loader_transforms = self.train_transform()

    train_dir = os.path.join(self.hparams.data_dir, "imagenet/train")
    train_dataset = ImageFolder(
      train_dir,
      loader_transforms,
    )

    train_loader = torch.utils.data.DataLoader(
      dataset=train_dataset,
      batch_size=self.hparams.batch_size,
      shuffle=self.train_shuffle,
      num_workers=self.hparams.num_workers
    )

    return train_loader

  def val_dataloader(self):
    loader_transforms = self.val_transform()

    val_dir = os.path.join(self.hparams.data_dir, "imagenet/val")

    val_loader = torch.utils.data.DataLoader(
      ImageFolder(
        val_dir,
        loader_transforms,
      ),
      batch_size=self.hparams.batch_size,
      shuffle=False,
      num_workers=self.hparams.num_workers,
    )
    return val_loader

  def test_dataloader(self):
    return self.val_dataloader()

  def train_transform(self) -> Callable:
      """The standard imagenet transforms.
      """
      normalize = transforms.Normalize(mean=self.mean, std=self.std)

      preprocessing = transforms.Compose(
          [
              transforms.RandomResizedCrop(self.image_size),
              transforms.RandomHorizontalFlip(),
              transforms.ToTensor(),
              normalize,
          ]
      )

      return preprocessing

  def val_transform(self) -> Callable:
      """The standard imagenet transforms for validation.
      """
      normalize = transforms.Normalize(mean=self.mean, std=self.std)

      preprocessing = transforms.Compose(
          [
              transforms.Resize(self.image_size + 32),
              transforms.CenterCrop(self.image_size),
              transforms.ToTensor(),
              normalize,
          ]
      )
      return preprocessing


class ImageNetDataIndexed(ImageNetData):
  def __init__(self, args):
    super().__init__(args)

  def train_dataloader(self):
    loader_transforms = self.train_transform()

    train_dir = os.path.join(self.hparams.data_dir, "train")
    train_dataset = ImageFolderMeta(
      train_dir,
      loader_transforms,
    )

    train_loader = torch.utils.data.DataLoader(
      dataset=train_dataset,
      batch_size=self.hparams.batch_size,
      shuffle=self.train_shuffle,
      num_workers=self.hparams.num_workers
    )

    return train_loader

  def val_dataloader(self):
    loader_transforms = self.val_transform()

    val_dir = os.path.join(self.hparams.data_dir, "val")

    val_loader = torch.utils.data.DataLoader(
      ImageFolderMeta(
        val_dir,
        loader_transforms,
      ),
      batch_size=self.hparams.batch_size,
      shuffle=False,
      num_workers=self.hparams.num_workers,
    )
    return val_loader


class ImageFolderMeta(ImageFolder):
  def __getitem__(self, index):
    path, target = self.samples[index]
    sample = self.loader(path)
    if self.transform is not None:
      sample = self.transform(sample)
    if self.target_transform is not None:
      target = self.target_transform(target)
    path = '/'.join(path.rsplit('/',2)[1:])
    return sample, target, path


class TinyImagenetData(pl.LightningDataModule):
  def __init__(self, args):
    super().__init__()
    self.hparams = args
    self.mean = (0.485, 0.456, 0.406)
    self.std = (0.229, 0.224, 0.225)

  def train_dataloader(self, shuffle=True, aug=True):
    """added optional shuffle parameter for generating random labels.
    added optional aug parameter to apply augmentation or not.

    Note: preprocessing is different for imagenet vs cifar10 models.
    """
    if aug is True:
      transform = transforms.Compose(
        [
          #transforms.Resize(256),
          #transforms.RandomResizedCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize(self.mean, self.std),
        ]
      )
    else:
      transform = transforms.Compose(
        [
          transforms.ToTensor(),
          transforms.Normalize(self.mean, self.std),
        ]
      )
    train_dir = os.path.join(self.hparams.data_dir, "train")
    train_dataset = ImageFolder(
      train_dir,
      transform,
    )

    if self.set_targets_train is not None:
      raise NotImplementedError('custom train targets NA')

    dataloader = torch.utils.data.DataLoader(
      train_dataset,
      batch_size=self.hparams.batch_size,
      num_workers=self.hparams.num_workers,
      shuffle=shuffle,
      drop_last=False,
      #pin_memory=True,
    )
    return dataloader

  def val_dataloader(self):
    val_dir = os.path.join(self.hparams.data_dir, "val")
    val_loader = torch.utils.data.DataLoader(
      ImageFolder(
        val_dir,
        transforms.Compose(
          [
           #transforms.Resize(256),
           #transforms.CenterCrop(224),
           transforms.ToTensor(),
           transforms.Normalize(self.mean, self.std),
           ]
        ),
      ),
      batch_size=self.hparams.batch_size,
      shuffle=False,
      num_workers=self.hparams.num_workers,
    )
    return val_loader

  def test_dataloader(self):
    return self.val_dataloader()