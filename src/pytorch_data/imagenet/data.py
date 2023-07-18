import os
from typing import Callable

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from ..utils import BaseDataModule

__all__lp = [
    "ImageNetData",
    "ImageNetDataIndexed",
    "TinyImagenetData"
]


class ImageFolderMeta(ImageFolder):
  # get specific image from folder
  def __getitem__(self, index):
    path, target = self.samples[index]
    sample = self.loader(path)
    if self.transform is not None:
      sample = self.transform(sample)
    if self.target_transform is not None:
      target = self.target_transform(target)
    path = '/'.join(path.rsplit('/',2)[1:])
    return sample, target, path


class ImageNetData(BaseDataModule):
  def __init__(self, args):
    super().__init__(args)
    self.mean = (0.485, 0.456, 0.406)
    self.std = (0.229, 0.224, 0.225)
    self.image_size = 224

    self.pin_memory = False

    self.train_shuffle = self.hparams.get("train_shuffle", True)
    if not self.train_shuffle:
      raise NotImplementedError("Need to implement train_noaug_dataset()")

  def setup(self, stage=None):

    train_dir = os.path.join(self.hparams.data_dir, "imagenet/train")
    val_dir = os.path.join(self.hparams.data_dir, "imagenet/val")

    train_set = ImageFolder(train_dir, self.train_transform())
    val_set = ImageFolder(train_dir, self.val_transform())
    test_set = ImageFolder(val_dir, self.val_transform())

    self._split_train_set(train_set, test_set, val_set)

  def train_transform(self, aug=True) -> Callable:
      """The standard imagenet transforms.
      """
      normalize = transforms.Normalize(mean=self.mean, std=self.std)

      if aug is True:
          preprocessing = transforms.Compose(
              [
                  transforms.RandomResizedCrop(self.image_size),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                  normalize,
              ]
          )

          return preprocessing
      else:
        return self.val_transform()

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

  def setup(self, stage=None):

    train_dir = os.path.join(self.hparams.data_dir, "imagenet/train")
    val_dir = os.path.join(self.hparams.data_dir, "imagenet/val")

    train_set = ImageFolderMeta(train_dir, self.train_transform())
    val_set = ImageFolderMeta(train_dir, self.val_transform())
    test_set = ImageFolderMeta(val_dir, self.val_transform())

    self._split_train_set(train_set, test_set, val_set)


class TinyImagenetData(BaseDataModule):
  def __init__(self, args):
    super().__init__(args)
    self.mean = (0.485, 0.456, 0.406)
    self.std = (0.229, 0.224, 0.225)
    self.pin_memory = False

  def setup(self, stage=None):

    train_dir = os.path.join(self.hparams.data_dir, "tiny-imagenet-200/train")
    val_dir = os.path.join(self.hparams.data_dir, "tiny-imagenet-200/val")

    train_set = ImageFolder(train_dir, self.train_transform())
    val_set = ImageFolder(train_dir, self.val_transform())
    test_set = ImageFolder(val_dir, self.val_transform())

    self._split_train_set(train_set, test_set, val_set)

  def train_transform(self, aug=True):
    if aug is True:
      transform = transforms.Compose(
        [
          # transforms.Resize(256),
          # transforms.RandomResizedCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize(self.mean, self.std),
        ]
      )
      return transform
    else:
      return self.val_transform()

  def val_transform(self):
      transform = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize(self.mean, self.std),
      ])
      return transform
