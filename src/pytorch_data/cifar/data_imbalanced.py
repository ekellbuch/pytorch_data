"""
Adopted from
https://github.com/kaidic/LDAM-DRW/blob/master/imbalance_cifar.py
"""
import numpy as np
from .data import CIFAR10Data, CIFAR100Data
from torchvision import transforms as T
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader
import pytorch_lightning as pl

__all__ = ["IMBALANCECIFAR10",
           "IMBALANCECIFAR100",
           ]

__all__lp = [
    "IMBALANCECIFAR10Data",
    "IMBALANCECIFAR100Data"
]


class IMBALANCECIFAR10(CIFAR10):
    cls_num = 10

    def __init__(self,
                 root,
                 imb_type='exp',  # exp, step
                 imb_factor=0.01,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False):
        super(IMBALANCECIFAR10, self).__init__(root, train, transform,
                                               target_transform, download)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type,
                                                imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
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
            new_targets.extend([
                the_class,
            ] * the_img_num)
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
        self.imb_type = self.hparams.get('imb_type', 'exp')
        self.imb_factor = self.hparams.get('imb_factor', 0.01)

    def get_torch_data_cls(self, train=True, aug=False):
        if train:
            transform = self.train_transform(aug=aug)
            dataset = IMBALANCECIFAR10(root=self.hparams.data_dir,
                               train=True,
                               transform=transform,
                               download=True,
                               imb_type=self.imb_type,
                               imb_factor=self.imb_factor
                                       )
        else:
            transform = self.valid_transform()
            dataset = CIFAR10(
                root=self.hparams.data_dir,
                train=False,
                transform=transform,
                download=True)
        return dataset


class IMBALANCECIFAR100(IMBALANCECIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    cls_num = 100
    dataset_name = 'CIFAR-100-LT'
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }


class IMBALANCECIFAR100Data(CIFAR100Data):
    def __init__(self, args):
        super().__init__(args)
        self.imb_type = self.hparams.get('imb_type', 'exp')  # imbalance type
        self.imb_factor = self.hparams.get('imb_factor',
                                           0.01)  # imbalance factor

    def get_torch_data_cls(self, train=True, aug=False):

        if train:
            transform = self.train_transform(aug=aug)
            dataset = IMBALANCECIFAR100(root=self.hparams.data_dir,
                               train=True,
                               transform=transform,
                               download=True,
                               imb_type=self.imb_type,
                               imb_factor=self.imb_factor
                                       )
        else:
            transform = self.valid_transform()
            dataset = CIFAR100(
                root=self.hparams.data_dir,
                train=False,
                transform=transform,
                download=True)
        return dataset