"""
Adopted from
https://github.com/kaidic/LDAM-DRW/blob/master/imbalance_cifar.py
"""
import numpy as np
import torch
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms as T
from .data import CIFAR10Data, CIFAR100Data
from pytorch_data.autoaugment import CIFAR10Policy, Cutout


__all__ = ["IMBALANCECIFAR10",
           "IMBALANCECIFAR100",
           ]

__all__lp = [
    "IMBALANCECIFAR10Data",  # cifar10-LT dataset
    "IMBALANCECIFAR10DataAug",  # cifar10-LT with cifar10 aug policy
    "IMBALANCECIFAR10DataAug2",  # cifar10-LT w cifar10 aug policy and rotation
    "IMBALANCECIFAR10Data_v1",  # cifar10-LT where val set is nonzero for any class
    "IMBALANCECIFAR10DataAug2_v1",  #
    "IMBALANCECIFAR100Data",  # cifar100-LT dataset
    "IMBALANCECIFAR100DataAug",  # cifar100-LT with cifar10 aug policy
    "IMBALANCECIFAR100DataAug_v1",  # imbalanced cifar 10 dataset where val set is nonzero for any class

]


class ValidationSetReassigner:
    def reassign_val_set(self, train_set, valid_set):
        # assign val dataset
        # val set is not in imbalanced_train_indices
        # reassign train dataset if valid_type is uniform or match
        extra_indices = np.asarray(list(set(range(len(valid_set))) - set(train_set.imbalanced_train_indices)))
        num_extra = len(extra_indices)

        split = min(int(np.floor(self.valid_size * len(train_set.imbalanced_train_indices))), len(extra_indices))
        if self.valid_type == 'random':
            # choose val set by selecting samples at random
            indices = torch.randperm(num_extra,
                                     generator=self.generator_from_seed,
                                     )[:split]

            self.val_indices = extra_indices[indices]
        elif self.valid_type == 'uniform':
            # the validation set has the same number of samples in each class (if available)
            extra_targets = np.asarray(valid_set.targets)[extra_indices]
            extra_class, extra_count = np.unique(extra_targets, return_counts=True)

            split_p_class = np.ceil(split / len(extra_class))
            # in each class (extra_count) there should be at least split_p_class
            assert all(extra_count >= split_p_class)
            val_indices = []
            for class_idx, class_id in enumerate(extra_class):
                class_mask = extra_targets == class_id
                class_extra_indices = extra_indices[class_mask]
                idx_to_sample = torch.randperm(int(split_p_class), generator=self.generator_from_seed)
                selec_idx = class_extra_indices[idx_to_sample]
                val_indices.append(selec_idx)

            val_indices = np.concatenate(val_indices)[:split].tolist()

            self.val_indices = val_indices

        elif self.valid_type == 'match':
            # the validation set has the same ratio as the train set for each class (if available)
            train_class, train_count = np.unique(train_set.targets, return_counts=True)
            split_p_class = np.ceil(self.valid_size * train_count)
            extra_targets = np.asarray(valid_set.targets)[extra_indices]
            extra_class, extra_count = np.unique(extra_targets, return_counts=True)
            # in each class (extra_count) there should be at least split_p_class
            val_indices = []
            for class_idx, class_id in enumerate(extra_class):
                class_mask = extra_targets == class_id
                class_extra_indices = extra_indices[class_mask]
                idx_to_sample = torch.randperm(int(split_p_class[class_id]), generator=self.generator_from_seed)
                selec_idx = class_extra_indices[idx_to_sample]
                val_indices.append(selec_idx)

            val_indices = np.concatenate(val_indices)[-split:].tolist()

            self.val_indices = val_indices

        self.val_dataset = Subset(valid_set, self.val_indices)


class IMBALANCECIFAR10(CIFAR10):
    cls_num = 10

    def __init__(self,
                 root,
                 imb_type='exp',  # exp, step
                 imb_factor=0.01,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False,
                 rand_number=0,
                 valid_size=0,):
        super(IMBALANCECIFAR10, self).__init__(root, train, transform,
                                               target_transform, download)
        if rand_number is not None:
            np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type,
                                                imb_factor)

        # if the validation size is > 0, adjust the number of samples per class
        # so that there are no classes with 0 samples in the validation set.
        if valid_size > 0:
            img_max = int(len(self.data) / self.cls_num)
            max_train_size = int(img_max - img_max * valid_size)
            img_num_list = [p_cls_num if p_cls_num <= max_train_size else max_train_size for p_cls_num in img_num_list]

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
        # keep track of indices
        new_indices = []
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
            new_indices.append(selec_idx)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        self.imbalanced_train_indices = np.concatenate(new_indices)

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


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



class IMBALANCECIFAR10Data(CIFAR10Data,ValidationSetReassigner):

    def __init__(self, args):
        super().__init__(args)
        self.imb_type = self.hparams.get('imb_type', 'exp')
        self.imb_factor = self.hparams.get('imb_factor', 0.01)
        self.valid_type = self.hparams.get('valid_type', 'random')

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        train_set = IMBALANCECIFAR10(self.hparams.data_dir,
                                     train=True,
                                     transform=self.train_transform(),
                                     imb_type=self.imb_type,
                                     imb_factor=self.imb_factor,
                                     rand_number=self.seed)
        # imbalanced_train_indices has the indices used for training:
        # use the remaining indices for validation.
        valid_set = CIFAR10(self.hparams.data_dir,
                           train=True,
                           transform=self.valid_transform())

        test_set = CIFAR10(self.hparams.data_dir,
                           train=False,
                           transform=self.valid_transform())

        assert self.valid_size < 1, "valid_size should be less than 1"
        assert self.valid_size >= 0, "valid_size should be greater than or eq to 0"

        self.train_dataset = train_set
        if self.valid_size == 0:
            self.val_dataset = test_set
        else:
            self.imbalanced_train_indices = train_set.imbalanced_train_indices
            self.reassign_val_set(train_set, valid_set)
        self.test_dataset = test_set

        self.check_targets()

    def train_noaug_dataset(self):
        # get dataset without transformations
        try:
            # compatible with some versions of lightning but not others
            train_set = CIFAR10(self.hparams.data_dir,
                                train=True,
                                transform=self.valid_transform(),
                                )

            return Subset(train_set, self.imbalanced_train_indices)
        except:
            raise NotImplementedError('No augmentation dataset not implemented for this dataset')

class IMBALANCECIFAR10DataAug(IMBALANCECIFAR10Data):
    """
    Compared to IMBALANCECIFAR10Data we use the CIFAR10Policy for augmentation.
    """
    def __init__(self, args):
        super().__init__(args)

    def train_transform(self, aug=True):
        if aug is True:
            transform = T.Compose([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                CIFAR10Policy(),    # add AutoAug
                T.ToTensor(),
                Cutout(n_holes=1, length=16),  # add Cutout
                T.Normalize(self.mean, self.std),
            ])
            return transform
        else:
            return self.valid_transform()


class IMBALANCECIFAR10DataAug2(IMBALANCECIFAR10Data):
    """
    Compared to IMBALANCECIFAR10Data we use the CIFAR10Policy for augmentation
    including a rotation augmentation
    """
    def __init__(self, args):
        super().__init__(args)

    def train_transform(self, aug=True):
        if aug is True:
            transform = T.Compose([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                CIFAR10Policy(),    # add AutoAug
                T.ToTensor(),
                T.RandomRotation(15),
                Cutout(n_holes=1, length=16),  # add Cutout
                T.Normalize(self.mean, self.std),
            ])
            return transform
        else:
            return self.valid_transform()


class IMBALANCECIFAR10Data_v1(IMBALANCECIFAR10Data):
    """
    Compared to IMBALANCECIFAR10Data we include 2 changes:
    # (1) include valid_size in the data constructor:
    #   this partitions the train data into train and val set
        where we leave some samples available in the top classes for the validation set.
    # (2) make validation set uniform across classes, i.e. balanced.
    """
    def __init__(self, args):
        super().__init__(args)
        self.valid_type = self.hparams.get('valid_type', 'uniform')

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        train_set = IMBALANCECIFAR10(self.hparams.data_dir,
                                     train=True,
                                     transform=self.train_transform(),
                                     imb_type=self.imb_type,
                                     imb_factor=self.imb_factor,
                                     rand_number=self.seed,
                                     valid_size=self.valid_size)
        # imbalanced_train_indices has the indices used for training:
        # use the remaining indices for validation.
        valid_set = CIFAR10(self.hparams.data_dir,
                           train=True,
                           transform=self.valid_transform())

        test_set = CIFAR10(self.hparams.data_dir,
                           train=False,
                           transform=self.valid_transform())

        assert self.valid_size < 1, "valid_size should be less than 1"
        assert self.valid_size >= 0, "valid_size should be greater than or eq to 0"

        self.train_dataset = train_set
        if self.valid_size == 0:
            self.val_dataset = test_set
        else:
            self.imbalanced_train_indices = train_set.imbalanced_train_indices
            self.reassign_val_set(train_set, valid_set)
        self.test_dataset = test_set

        self.check_targets()


class IMBALANCECIFAR10DataAug2_v1(IMBALANCECIFAR10Data_v1):
    # compared to IMBALANCECIFAR10Data_v1 adds rotation to augmentation.
    def __init__(self, args):
        super().__init__(args)

    def train_transform(self, aug=True):
        if aug is True:
            transform = T.Compose([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                CIFAR10Policy(),  # add AutoAug
                T.ToTensor(),
                T.RandomRotation(15),
                Cutout(n_holes=1, length=16),  # add Cutout
                T.Normalize(self.mean, self.std),
            ])
            return transform
        else:
            return self.valid_transform()


class IMBALANCECIFAR100Data(CIFAR100Data,ValidationSetReassigner):
    def __init__(self, args):
        super().__init__(args)
        self.imb_type = self.hparams.get('imb_type', 'exp')  # imbalance type
        self.imb_factor = self.hparams.get('imb_factor', 0.01)  # imbalance factor
        self.valid_type = self.hparams.get('valid_type', 'random')

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        train_set = IMBALANCECIFAR100(self.hparams.data_dir,
                                      train=True,
                                      transform=self.train_transform(),
                                      imb_type=self.imb_type,
                                      imb_factor=self.imb_factor,
                                      rand_number=self.seed)

        valid_set = CIFAR100(self.hparams.data_dir,
                           train=True,
                           transform=self.valid_transform())


        test_set = CIFAR100(self.hparams.data_dir, train=False, transform=self.valid_transform())

        assert self.valid_size < 1, "valid_size should be less than 1"
        assert self.valid_size >= 0, "valid_size should be greater than or eq to 0"

        self.train_dataset = train_set
        if self.valid_size == 0:
            self.val_dataset = test_set
        else:
            self.imbalanced_train_indices = train_set.imbalanced_train_indices
            self.reassign_val_set(train_set, valid_set)
        self.test_dataset = test_set

        self.check_targets()

    def train_noaug_dataset(self):
        # get dataset without transformations
        try:
            # compatible with some versions of lightning but not others
            train_set = CIFAR100(self.hparams.data_dir,
                                train=True,
                                transform=self.valid_transform(),
                                )

            return Subset(train_set, self.imbalanced_train_indices)
        except:
            raise NotImplementedError('No augmentation dataset not implemented for this dataset')


class IMBALANCECIFAR100DataAug(IMBALANCECIFAR100Data):

    def __init__(self, args):
        super().__init__(args)

    def train_transform(self, aug=True):
        if aug is True:
            transform = T.Compose([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                CIFAR10Policy(),    # add AutoAug
                T.ToTensor(),
                Cutout(n_holes=1, length=16),  # add Cutout
                T.Normalize(self.mean, self.std),
            ])
            return transform
        else:
            return self.valid_transform()


class IMBALANCECIFAR100Data_v1(IMBALANCECIFAR100Data):
    """
    same changes as IMBALANCECIFAR10Data_v1
    """
    def __init__(self, args):
        super().__init__(args)
        self.valid_type = self.hparams.get('valid_type', 'uniform')

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        train_set = IMBALANCECIFAR100(self.hparams.data_dir,
                                     train=True,
                                     transform=self.train_transform(),
                                     imb_type=self.imb_type,
                                     imb_factor=self.imb_factor,
                                     rand_number=self.seed,
                                     valid_size=self.valid_size)
        # imbalanced_train_indices has the indices used for training:
        # use the remaining indices for validation.
        valid_set = CIFAR100(self.hparams.data_dir,
                           train=True,
                           transform=self.valid_transform())

        test_set = CIFAR100(self.hparams.data_dir,
                           train=False,
                           transform=self.valid_transform())

        assert self.valid_size < 1, "valid_size should be less than 1"
        assert self.valid_size >= 0, "valid_size should be greater than or eq to 0"

        self.train_dataset = train_set
        if self.valid_size == 0:
            self.val_dataset = test_set
        else:
            self.imbalanced_train_indices = train_set.imbalanced_train_indices
            self.reassign_val_set(train_set, valid_set)
        self.test_dataset = test_set

        self.check_targets()

class IMBALANCECIFAR100DataAug_v1(IMBALANCECIFAR100Data_v1):
    # compared to IMBALANCECIFAR100Data_v1 adds cifar10 augmentation policy.
    def __init__(self, args):
        super().__init__(args)

    def train_transform(self, aug=True):
        if aug is True:
            transform = T.Compose([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                CIFAR10Policy(),  # add AutoAug
                T.ToTensor(),
                Cutout(n_holes=1, length=16),  # add Cutout
                T.Normalize(self.mean, self.std),
            ])
            return transform
        else:
            return self.valid_transform()