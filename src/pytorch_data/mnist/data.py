## smaller datasets that aren't cifar10
import pytorch_lightning as pl
import requests
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST


class OneHotTransform():
  def __init__(self, num_classes):
    self.num_classes = num_classes

  def __call__(self, data):
    return torch.nn.functional.one_hot(torch.tensor(data), self.num_classes).float()


class MNISTModule(pl.LightningDataModule):
  def __init__(self, args):
    super().__init__()
    self.hparams = args
    self.mnist_predict = MNIST(self.hparams.data_dir, train=False, transform=ToTensor(),
                               target_transform=OneHotTransform(num_classes=10), download=True)
    self.mnist_train = MNIST(self.hparams.data_dir, train=True, transform=ToTensor(),
                             target_transform=OneHotTransform(num_classes=10), download=True)

  def train_dataloader(self, shuffle=False, aug=False):
    return DataLoader(self.mnist_train,
                      batch_size=self.hparams.batch_size,
                      num_workers=self.hparams.num_workers,
                      drop_last=False,
                      pin_memory=True,
                      shuffle=shuffle)

  def val_dataloader(self):
    return DataLoader(self.mnist_predict,
                      batch_size=self.hparams.batch_size,
                      num_workers=self.hparams.num_workers,
                      drop_last=False,
                      pin_memory=True, )

  def test_dataloader(self):
    return self.val_dataloader()