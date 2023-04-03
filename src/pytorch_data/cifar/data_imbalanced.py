import numpy as np
from .data import CIFAR10Data, CIFAR100Data
from torchvision import transforms as T
import torchvision
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader


class IMBALANCECIFAR10(CIFAR10):
  # from https://github.com/kaidic/LDAM-DRW/blob/master/imbalance_cifar.py
  cls_num = 10

  def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
               transform=None, target_transform=None,
               download=False):
    super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
    np.random.seed(rand_number)
    img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
    self.gen_imbalanced_data(img_num_list)

  def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
    img_max = len(self.data) / cls_num
    img_num_per_cls = []
    if imb_type == 'exp':
      for cls_idx in range(cls_num):
        num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
        img_num_per_cls.append(int(num))
    elif imb_type == 'step':
      for cls_idx in range(cls_num // 2):
        img_num_per_cls.append(int(img_max))
      for cls_idx in range(cls_num // 2):
        img_num_per_cls.append(int(img_max * imb_factor))
    else:
      img_num_per_cls.extend([int(img_max)] * cls_num)
    return img_num_per_cls

  def gen_imbalanced_data(self, img_num_per_cls):
    new_data = []
    new_targets = []
    targets_np = np.array(self.targets, dtype=np.int64)
    classes = np.unique(targets_np)
    # np.random.shuffle(classes)
    self.num_per_cls_dict = dict()
    for the_class, the_img_num in zip(classes, img_num_per_cls):
      self.num_per_cls_dict[the_class] = the_img_num
      idx = np.where(targets_np == the_class)[0]
      np.random.shuffle(idx)
      selec_idx = idx[:the_img_num]
      new_data.append(self.data[selec_idx, ...])
      new_targets.extend([the_class, ] * the_img_num)
    new_data = np.vstack(new_data)
    self.data = new_data
    self.targets = new_targets

  def get_cls_num_list(self):
    cls_num_list = []
    for i in range(self.cls_num):
      cls_num_list.append(self.num_per_cls_dict[i])
    return cls_num_list


class IMBALANCECIFAR10Data(CIFAR10Data):
  def __init__(self, args):
    super().__init__(args)
    #self.hparams = args
    self.imb_type = self.hparams.get('imb_type', 'exp')   # imbalance type
    self.imb_factor = self.hparams.get('imb_factor', 0.01)  # imbalance factor
    self.rand_number = self.hparams.get('seed', 0)  # random seed
    self.cls_num = self.hparams.get('cls_num', 100)  # number of classes

  def train_dataloader(self, shuffle=True, aug=True):
    """added optional shuffle parameter for generating random labels.
    added optional aug parameter to apply augmentation or not.

    """
    if aug is True:
      transform = T.Compose(
        [
          T.RandomCrop(32, padding=4),
          T.RandomHorizontalFlip(),
          T.ToTensor(),
          T.Normalize(self.mean, self.std),
        ]
      )
    else:
      transform = T.Compose(
        [
          T.ToTensor(),
          T.Normalize(self.mean, self.std),
        ]
      )
    dataset = IMBALANCECIFAR10(root=self.hparams.data_dir, train=True, transform=transform, download=False,
                               imb_type=self.imb_type, imb_factor=self.imb_factor, rand_number=self.rand_number)

    if self.set_targets_train is not None:
      dataset.targets = self.set_targets_train
      assert len(dataset.data) == len(dataset.targets), "number of examples, {} does not match targets {}".format(
        len(dataset.data), len(dataset.targets))
      assert dataset.data.shape[1] >= np.max(
        dataset.targets), "number of classes, {} does not match target index {}".format(dataset.data.shape[1],
                                                                                        np.max(dataset.targets))

    dataloader = DataLoader(
      dataset,
      batch_size=self.hparams.batch_size,
      num_workers=self.hparams.num_workers,
      shuffle=shuffle,
      drop_last=False,
      pin_memory=True,
    )
    return dataloader


class IMBALANCECIFAR100(CIFAR100):
  # from https://github.com/kaidic/LDAM-DRW/blob/master/imbalance_cifar.py
  cls_num = 100

  def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
               transform=None, target_transform=None,
               download=False):
    super(IMBALANCECIFAR100, self).__init__(root, train, transform, target_transform, download)
    np.random.seed(rand_number)
    img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
    self.gen_imbalanced_data(img_num_list)

  def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
    img_max = len(self.data) / cls_num
    img_num_per_cls = []
    if imb_type == 'exp':
      for cls_idx in range(cls_num):
        num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
        img_num_per_cls.append(int(num))
    elif imb_type == 'step':
      for cls_idx in range(cls_num // 2):
        img_num_per_cls.append(int(img_max))
      for cls_idx in range(cls_num // 2):
        img_num_per_cls.append(int(img_max * imb_factor))
    else:
      img_num_per_cls.extend([int(img_max)] * cls_num)
    return img_num_per_cls

  def gen_imbalanced_data(self, img_num_per_cls):
    new_data = []
    new_targets = []
    targets_np = np.array(self.targets, dtype=np.int64)
    classes = np.unique(targets_np)
    # np.random.shuffle(classes)
    self.num_per_cls_dict = dict()
    for the_class, the_img_num in zip(classes, img_num_per_cls):
      self.num_per_cls_dict[the_class] = the_img_num
      idx = np.where(targets_np == the_class)[0]
      np.random.shuffle(idx)
      selec_idx = idx[:the_img_num]
      new_data.append(self.data[selec_idx, ...])
      new_targets.extend([the_class, ] * the_img_num)
    new_data = np.vstack(new_data)
    self.data = new_data
    self.targets = new_targets

  def get_cls_num_list(self):
    cls_num_list = []
    for i in range(self.cls_num):
      cls_num_list.append(self.num_per_cls_dict[i])
    return cls_num_list


class IMBALANCECIFAR100Data(CIFAR100Data):
  def __init__(self, args):
    super().__init__(args)
    self.imb_type = self.hparams.get('imb_type', 'exp')   # imbalance type
    self.imb_factor = self.hparams.get('imb_factor', 0.01)  # imbalance factor
    self.rand_number = self.hparams.get('seed', 0)  # random seed
    self.cls_num = self.hparams.get('cls_num', 100)  # number of classes

  def train_dataloader(self, shuffle=True, aug=True):
    """added optional shuffle parameter for generating random labels.
    added optional aug parameter to apply augmentation or not.

    """
    if aug is True:
      transform = T.Compose(
        [
          T.RandomCrop(32, padding=4),
          T.RandomHorizontalFlip(),
          T.ToTensor(),
          T.Normalize(self.mean, self.std),
        ]
      )
    else:
      transform = T.Compose(
        [
          T.ToTensor(),
          T.Normalize(self.mean, self.std),
        ]
      )
    dataset = IMBALANCECIFAR100(root=self.hparams.data_dir, train=True, transform=transform, download=False,
                               imb_type=self.imb_type, imb_factor=self.imb_factor, rand_number=self.rand_number)

    if self.set_targets_train is not None:
      dataset.targets = self.set_targets_train
      assert len(dataset.data) == len(dataset.targets), "number of examples, {} does not match targets {}".format(
        len(dataset.data), len(dataset.targets))
      assert dataset.data.shape[1] >= np.max(
        dataset.targets), "number of classes, {} does not match target index {}".format(dataset.data.shape[1],
                                                                                        np.max(dataset.targets))

    dataloader = DataLoader(
      dataset,
      batch_size=self.hparams.batch_size,
      num_workers=self.hparams.num_workers,
      shuffle=shuffle,
      drop_last=False,
      pin_memory=True,
    )
    return dataloader
