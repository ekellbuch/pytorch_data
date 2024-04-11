import os
import zipfile
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T
from torchvision.datasets import CIFAR10, CIFAR100
from tqdm import tqdm
from pytorch_data.autoaugment import CIFAR10Policy, Cutout

from ..utils import stream_download, BaseDataModule

__all__ = [
    "CIFAR10_C",
    "CINIC10",
    "CIFAR10_1",
    "CIFAR100Coarse",
]

__all__lp = [
     "CIFAR10Data",
     "CIFAR10DatatAug_v2",  # adds cutout policy for cifar 10
     "CINIC10_Data",
     "CIFAR10_1Data",
     "CIFAR10_CData",
     "CIFAR100Data",
     "CIFAR100CoarseData"
]


# val set sample size per class
NUM_SAMPLES_TEST = {
  "cifar10": 1000,
  "cifar100": 100,
  "imagenet": 50,
}

NUM_SAMPLES_TRAIN = {
  "cifar10": 6000,
  "cifar100": 500,
}




def unzip_to(source, target_directory):
    """Unzips source to create a folder in the target directory

  """
    with zipfile.ZipFile(source, "r") as zf:
        for member in tqdm(zf.infolist(), desc='Extracting '):
            try:
                zf.extract(member, target_directory)
            except zipfile.error as e:
                print("Bad zipfile: {0}".format(e), flush=True)
                pass


class CIFAR10_C(torchvision.datasets.vision.VisionDataset):
    """Pytorch wrapper around the CIFAR10-C dataset (https://zenodo.org/record/2535967#.YbDe8i1h3PA)

  """

    def __len__(self) -> int:
        pass

    def __getitem__(self, index: int) -> Any:
        pass

    def __init__(self,
                 root_dir,
                 corruption,
                 level,
                 transform=None,
                 target_transform=None):
        """
    :param root_dir: path to store data files.
    :param corruption: the kind of corruption we want to study. should be one of:
      - brightness
      - contrast
      - defocus_blur
      - elastic_transform
      - fog
      - frost
      - gaussian_blur
      - gaussian_noise
      - glass_blur
      - impulse_noise
      - jpeg_compression
      - motion_blur
      - pixelate
      - saturate
      - shot_noise
      - snow
      - spatter
      - speckle_noise
      - zoom_blur
    :param level: a corruption level from 1-5.
    :param transform: an optional transform to be applied on PIL image samples.
    :param target_transform: an optional transform to be applied to targets.
    **NB: there is an attribute "transforms" that we don't make use of in this class that the original CIFAR10 might.**
    """
        here = os.path.dirname(os.path.abspath(__file__))
        super().__init__(root_dir,
                         transform=transform,
                         target_transform=target_transform)
        assert corruption in [
            "brightness",
            "contrast",
            "defocus_blur",
            "elastic_transform",
            "fog",
            "frost",
            "gaussian_blur",
            "gaussian_noise",
            "glass_blur",
            "impulse_noise",
            "jpeg_compression",
            "motion_blur",
            "pixelate",
            "saturate",
            "shot_noise",
            "snow",
            "spatter",
            "speckle_noise",
            "zoom_blur",
        ], "corruption must be one of those listed. (check docstring)"
        self.corruption = corruption
        assert level in np.arange(1, 6), "level must be between 1 and 5 "
        self.level = level
        # Check data unzipped:
        corruption_dir = os.path.join(root_dir, self.corruption)
        if not os.path.exists(corruption_dir):
            os.mkdir(corruption_dir)
        data_path = os.path.join(
            corruption_dir, "cifar10c_{}_data.npy".format(self.corruption))
        target_path = os.path.join(
            corruption_dir, "cifar10c_{}_labels.npy".format(self.corruption))
        if not os.path.exists(data_path) or not os.path.exists(target_path):
            zippath = os.path.join(
                here, "../../data/cifar10-c/{}.zip".format(self.corruption))
            assert os.path.exists(zippath)
            "Error: zipped cifar10-c dataset not located."

            unzip_to(zippath, corruption_dir
                     )  # creates cinic-10 dataset at requested location.

        # Now get data and put it in memory:
        self.alldata = np.load(data_path)
        self.all_targets = list(np.load(target_path))
        start_index = (self.level - 1) * 10000
        end_index = self.level * 10000
        self.data = self.alldata[start_index:end_index, :]
        self.targets = self.all_targets[start_index:end_index]
        # Class-index mapping from original CIFAR10 dataset:
        self.classes = [
            'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
            'horse', 'ship', 'truck'
        ]
        self.class_to_idx = {
            'airplane': 0,
            'automobile': 1,
            'bird': 2,
            'cat': 3,
            'deer': 4,
            'dog': 5,
            'frog': 6,
            'horse': 7,
            'ship': 8,
            'truck': 9
        }


class CINIC10(torchvision.datasets.ImageFolder):
    """Pytorch wrapper around the CINIC10 (imagenet only) dataset (https://github.com/BayesWatch/cinic-10). Assumes
  that the zipped dataset is already stored locally () using git LFS.

  """

    def __init__(self,
                 root_dir,
                 split,
                 transform=None,
                 target_transform=None,
                 loader=None,
                 is_valid_file=None,
                 preload=False,
                 labels=None):
        """
    :param root_dir: path to store data files.
    :param split: which split (train/test/val) of the data to grab.
    :param transform: an optional transform to be applied on PIL image samples.
    :param target_transform: an optional transform to be applied to targets.
    :param loader: an optional callable to load in images.
    :param is_valid_file: an optional loader to check image validity.
    :param preload: optionally preload all images into .data attribute
    :param labels: if labels are given, use these instead of class labels
    """
        assert str(root_dir).endswith(
            "cinic-10"), "the directory must be called `cinic-10`"
        assert split in ["train", "test", "val"]
        self.labels = labels
        if not os.path.exists(os.path.join(root_dir, "train")):
            here = os.path.dirname(os.path.abspath(__file__))
            zippath = os.path.join(here, "../../data/cinic-10.zip")
            assert os.path.exists(zippath)
            "Error: zipped cinic-10 dataset not located."
            unzip_to(zippath, os.path.dirname(
                root_dir))  # creates cinic-10 dataset at requested location.

        # First, check if the dataset exists. If not, unzip it from local.
        args = {
            "transform": transform,
            "target_transform": target_transform,
            "loader": loader,
            "is_valid_file": is_valid_file
        }
        kws = {}
        for key in args:
            if args[key] is not None:
                kws[key] = args[key]
        super().__init__(os.path.join(root_dir, split), **kws)

        # Class-index mapping from original CIFAR10 dataset:
        self.classes = [
            'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
            'horse', 'ship', 'truck'
        ]
        self.class_to_idx = {
            'airplane': 0,
            'automobile': 1,
            'bird': 2,
            'cat': 3,
            'deer': 4,
            'dog': 5,
            'frog': 6,
            'horse': 7,
            'ship': 8,
            'truck': 9
        }
        if preload:
            self.data = np.stack([
                np.array(self.__getitem__(i)[0])
                for i in range(len(self.samples))
            ],
                                 axis=0)
            self.targets = [s[1] for s in self.samples]

    def __getitem__(self, idx):
        """
    optional override of getitem.
    """
        if self.labels is not None:
            img, _ = super().__getitem__(idx)
            label = self.labels[idx]
            return img, label
        else:
            return super().__getitem__(idx)


class CIFAR10_1(torchvision.datasets.vision.VisionDataset):
    """Pytorch wrapper around the CIFAR10.1 dataset (https://github.com/modestyachts/CIFAR-10.1)

  """

    def __init__(self,
                 root_dir,
                 version="v6",
                 transform=None,
                 train=False,
                 target_transform=None):
        """
    :param root_dir: path to store data files.
    :param version: there is a v4 and v6 version of this data.
    :param transform: an optional transform to be applied on PIL image samples.
    :param target_transform: an optional transform to be applied to targets.
    **NB: there is an attribute "transforms" that we don't make use of in this class that the original CIFAR10 might.**
    """
        super().__init__(root_dir,
                         transform=transform,
                         target_transform=target_transform)
        # self.root = root_dir
        self.version = version
        # self.transform = transform
        self.train = train  # should always be false.
        # self.target_transform = target_transform
        # self.transforms = None ## don't know what this parameter is.
        assert self.version in ["v4", "v6"]
        # Download data
        self.data_path, self.target_path = self.download()
        # Now get data and put it in memory:
        self.data = np.load(self.data_path)
        self.targets = list(np.load(self.target_path))
        # Class-index mapping from original CIFAR10 dataset:
        self.classes = [
            'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
            'horse', 'ship', 'truck'
        ]
        self.class_to_idx = {
            'airplane': 0,
            'automobile': 1,
            'bird': 2,
            'cat': 3,
            'deer': 4,
            'dog': 5,
            'frog': 6,
            'horse': 7,
            'ship': 8,
            'truck': 9
        }

    def download(self):
        """Download data from gitHub.

    :returns: path where the data and label are located.
    """
        root_path = "https://raw.github.com/modestyachts/CIFAR-10.1/master/datasets/"
        data_name = "cifar10.1_{}_data.npy".format(self.version)
        label_name = "cifar10.1_{}_labels.npy".format(self.version)

        dataurl = os.path.join(root_path, data_name)
        data_destination = os.path.join(self.root, "data.npy")
        if not os.path.exists(data_destination):
            stream_download(dataurl, data_destination)
        label_url = os.path.join(root_path, label_name)
        label_destination = os.path.join(self.root, "labels.npy")
        if not os.path.exists(label_destination):
            stream_download(label_url, label_destination)
        return data_destination, label_destination

    def __len__(self):
        """Get dataset length:

    """
        return len(self.targets)

    def __getitem__(self, idx):
        """Get an item from the dataset:

    """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = Image.fromarray(self.data[idx, :, :, :])
        target = np.int64(self.targets[idx])
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        sample = (img, target)
        return sample



class CINIC10_Data(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.hparams.update(args)
        self.mean = (0.47889522, 0.47227842, 0.43047404)
        self.std = (0.24205776, 0.23828046, 0.25874835)

        if args.get("custom_targets_eval_ood", False):
            self.set_targets_eval_ood = np.load(args.custom_targets_eval_ood)
        else:
            self.set_targets_eval_ood = None

    def check_targets(self, dataset):
        if self.set_targets_eval_ood is not None:
            dataset.targets = self.set_targets_eval_ood
            assert len(dataset.data) == len(
                dataset.targets
            ), "number of examples, {} does not match targets {}".format(
                len(dataset.data), len(dataset.targets))
            assert dataset.data.shape[1] >= np.max(
                dataset.targets
            ), "number of classes, {} does not match target index {}".format(
                dataset.data.shape[1], np.max(dataset.targets))
        return dataset

    def setup(self, stage=None):
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(self.mean, self.std),
        ])

        val_dataset = CINIC10(root_dir=os.path.join(self.hparams.data_dir, "cinic-10"), split="val",
                              transform=transform)
        self.val_dataset = self.check_targets(val_dataset)

        test_dataset = CINIC10(root_dir=os.path.join(self.hparams.data_dir, "cinic-10"), split="test",
                               transform=transform, labels=self.set_targets_eval_ood)
        self.test_dataset = self.check_targets(test_dataset)


    def train_dataloader(self):
        raise NotImplementedError(
            "don't know the right transforms for this- will implement later")


    def val_dataloader(self):
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=False,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=False,
            pin_memory=True,
        )
        return dataloader


class CIFAR10_1Data(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.hparams.update(args)  # check these.
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2471, 0.2435, 0.2616)
        self.version = args.get("version", "v6")

        if args.get("custom_targets_eval_ood", False):
            self.set_targets_eval_ood = np.load(args.custom_targets_eval_ood)
        else:
            self.set_targets_eval_ood = None

    def setup(self, stage=None):
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(self.mean, self.std),
        ])
        dataset = CIFAR10_1(root_dir=self.hparams.data_dir,
                            train=False,
                            transform=transform,
                            version=self.version)
        if self.set_targets_eval_ood is not None:
            dataset.targets = self.set_targets_eval_ood
            assert len(dataset.data) == len(
                dataset.targets
            ), "number of examples, {} does not match targets {}".format(
                len(dataset.data), len(dataset.targets))
            assert dataset.data.shape[1] >= np.max(
                dataset.targets
            ), "number of classes, {} does not match target index {}".format(
                dataset.data.shape[1], np.max(dataset.targets))
        self.val_dataset = dataset

    def train_dataloader(self):
        raise NotImplementedError

    def val_dataloader(self):
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=False,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()


class CIFAR10_CData(pl.LightningDataModule):
    """Dataset management- downloading, hyperparams, etc.

    :param args: the argparse parse_args output. key here is the root_dir parameter.
     we will work with a subdirectory of `root_dir` called cifar-10c .
    """

    def __init__(self, args):
        super().__init__()
        self.hparams.update(args)
        self.mean = (0.4914, 0.4822, 0.4465)  # should we revise these?
        self.std = (0.2471, 0.2435, 0.2616)  #
        if args.get("custom_targets_eval_ood", False):
            self.set_targets_eval_ood = np.load(args.custom_targets_eval_ood)
        else:
            self.set_targets_eval_ood = None

    def train_dataloader(self):
        raise NotImplementedError

    def val_dataloader(self):
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(self.mean, self.std),
        ])
        dataset = CIFAR10_C(root_dir=os.path.join(self.hparams.data_dir,
                                                  "cifar10-c"),
                            corruption=self.hparams.corruption,
                            level=self.hparams.level,
                            transform=transform)
        if self.set_targets_eval_ood is not None:
            dataset.targets = self.set_targets_eval_ood
            assert len(dataset.data) == len(
                dataset.targets
            ), "number of examples, {} does not match targets {}".format(
                len(dataset.data), len(dataset.targets))
            assert dataset.data.shape[1] >= np.max(
                dataset.targets
            ), "number of classes, {} does not match target index {}".format(
                dataset.data.shape[1], np.max(dataset.targets))

        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=False,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()


class CIFAR10Data(BaseDataModule):

    def __init__(self, args):
        super().__init__(args)
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2023, 0.1994, 0.2010)

        # if softmax targets are given, parse.
        if args.get("custom_targets_train", False):
            # self.set_targets_train = parse_softmax(args.softmax_targets_train)
            # training targets should be softmax! others should be binary.
            self.set_targets_train = np.load(args.custom_targets_train)
        else:
            self.set_targets_train = None
        if args.get("custom_targets_eval_ind", False):
            self.set_targets_eval_ind = np.load(args.custom_targets_eval_ind)
        else:
            self.set_targets_eval_ind = None

    @staticmethod
    def download_weights():
        url = (
            "https://rutgers.box.com/shared/static/gkw08ecs797j2et1ksmbg1w5t3idf5r5.zip"
        )

        stream_download(url, "state_dicts.zip")
        print("Download successful. Unzipping file...")
        path_to_zip_file = os.path.join(os.getcwd(), "state_dicts.zip")
        directory_to_extract_to = os.path.join(os.getcwd(), "cifar10_models")
        with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
            zip_ref.extractall(directory_to_extract_to)
            print("Unzip file successful!")

    def prepare_data(self):
        # download
        CIFAR10(self.hparams.data_dir, train=True, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        train_set = CIFAR10(self.hparams.data_dir,
                            train=True,
                            transform=self.train_transform())
        valid_set = CIFAR10(self.hparams.data_dir,
                            train=True,
                            transform=self.valid_transform())
        test_set = CIFAR10(self.hparams.data_dir,
                           train=False,
                           transform=self.valid_transform())

        self._split_train_set(train_set, test_set, valid_set)

        self.check_targets()

    def train_noaug_dataset(self):
        train_set = CIFAR10(self.hparams.data_dir,
                            train=True,
                            transform=self.valid_transform())
        if self.valid_size == 0:
            return train_set
        return Subset(train_set, self.train_indices)

    def check_targets(self):
        if self.set_targets_train is not None:
            self.train_dataset.targets = self.set_targets_train
            assert len(self.train_dataset.data) == len(
                self.train_dataset.targets
            ), "number of examples, {} does not match targets {}".format(
                len(self.train_dataset.data), len(self.train_dataset.targets))
            assert self.train_dataset.data.shape[1] >= np.max(
                self.train_dataset.targets
            ), "number of classes, {} does not match target index {}".format(
                self.train_dataset.data.shape[1], np.max(self.train_dataset.targets))

    def train_transform(self, aug=True):
        if aug is True:
            transform = T.Compose([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ])
            return transform
        else:
            return self.valid_transform()

    def valid_transform(self):
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(self.mean, self.std),
        ])
        return transform

class CIFAR10DatatAug_v2(CIFAR10Data):
    # compared to CIFAR10Data adds augmentations.
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


class CIFAR100Data(CIFAR10Data):
    def __init__(self, args):
        super().__init__(args)

        # hack to debug augmentation:
        if self.hparams.get('train_aug', None) is not None:
            raise ValueError("train_aug should not be set for CIFAR100")

    def prepare_data(self):
        # download CIFAR 100 dataset
        CIFAR100(self.hparams.data_dir, train=True, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        train_set = CIFAR100(self.hparams.data_dir, train=True, transform=self.train_transform())
        valid_set = CIFAR100(self.hparams.data_dir, train=True, transform=self.valid_transform())
        test_set = CIFAR100(self.hparams.data_dir, train=False, transform=self.valid_transform())

        self._split_train_set(train_set, test_set, valid_set)
        self.check_targets()

    def train_noaug_dataset(self):
        train_set = CIFAR100(self.hparams.data_dir, train=True, transform=self.valid_transform())
        if self.valid_size == 0:
            return train_set
        return Subset(train_set, self.train_indices)


class CIFAR100Coarse(CIFAR100):
    """CIFAR100 with coarse labels.
  from https://github.com/ryanchankh/cifar100coarse/blob/master/cifar100coarse.py
  """

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False):
        super(CIFAR100Coarse, self).__init__(root, train, transform,
                                             target_transform, download)

        # update labels
        coarse_labels = np.array([
            4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
            6, 11, 5, 10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10, 12, 14, 16, 9, 11,
            5, 5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0, 17, 4,
            18, 17, 10, 3, 2, 12, 12, 16, 12, 1, 9, 19, 2, 10, 0, 1, 16, 12, 9,
            13, 15, 13, 16, 19, 2, 4, 6, 19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0,
            17, 8, 14, 13
        ])
        self.targets = coarse_labels[self.targets]

        # update classes
        self.classes = [
            ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
            ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
            ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
            ['bottle', 'bowl', 'can', 'cup', 'plate'],
            ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
            ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
            ['bed', 'chair', 'couch', 'table', 'wardrobe'],
            ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
            ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
            ['bridge', 'castle', 'house', 'road', 'skyscraper'],
            ['cloud', 'forest', 'mountain', 'plain', 'sea'],
            ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
            ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
            ['crab', 'lobster', 'snail', 'spider', 'worm'],
            ['baby', 'boy', 'girl', 'man', 'woman'],
            ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
            ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
            [
                'maple_tree', 'oak_tree', 'palm_tree', 'pine_tree',
                'willow_tree'
            ], ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
            ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']
        ]


class CIFAR100CoarseData(CIFAR100Data):
    def __init__(self, args):
        super().__init__(args)

    def prepare_data(self):
        # download
        CIFAR100Coarse(self.hparams.data_dir, train=True, download=True)
        CIFAR100Coarse(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        train_set = CIFAR100Coarse(self.hparams.data_dir, train=True, transform=self.train_transform())
        valid_set = CIFAR100Coarse(self.hparams.data_dir, train=True, transform=self.valid_transform())
        test_set = CIFAR100Coarse(self.hparams.data_dir, train=False, transform=self.valid_transform())

        self._split_train_set(train_set, test_set, valid_set)
        self.check_targets()

    def train_noaug_dataset(self):
        train_set = CIFAR100Coarse(self.hparams.data_dir, train=True, transform=self.valid_transform())
        if self.valid_size == 0:
            return train_set
        return Subset(train_set, self.train_indices)