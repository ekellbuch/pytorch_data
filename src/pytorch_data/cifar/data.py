import os
import zipfile

from PIL import Image
import numpy as np
import pytorch_lightning as pl
import requests
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
import torchvision
from torchvision.datasets import CIFAR10, CIFAR100
from tqdm import tqdm


# def parse_softmax(path):
#    """Parse the numpy softmax outputs and convert them to a list of labels
#
#    :param path: path to a numpy file containing softmax outputs we can treat as labels.
#    """
#    softmax = np.load(path)
#    return list(np.argmax(np.load(path),axis=1))

def stream_download(dataurl, download_path):
  """helper function to monitor downloads.
  :param dataurl: path where data is located.
  :param download_path: local path (include filename) where we should write data.

  """
  r = requests.get(dataurl, stream=True)
  total_size = int(r.headers.get("content-length", 0))
  block_size = 2 ** 20  # Mebibyte
  t = tqdm(total=total_size, unit="MiB", unit_scale=True)

  with open(download_path, "wb") as f:
    for data in r.iter_content(block_size):
      t.update(len(data))
      f.write(data)
  t.close()


def unzip_to(source, targetdirectory):
  """Unzips source to create a folder in the target directory

  """
  with zipfile.ZipFile(source, "r") as zf:
    for member in tqdm(zf.infolist(), desc='Extracting '):
      try:
        zf.extract(member, targetdirectory)
      except zipfile.error as e:
        pass


class CIFAR10_C(torchvision.datasets.vision.VisionDataset):
  """Pytorch wrapper around the CIFAR10-C dataset (https://zenodo.org/record/2535967#.YbDe8i1h3PA)

  """

  def __init__(self, root_dir, corruption, level, transform=None, target_transform=None):
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
    super().__init__(root_dir, transform=transform, target_transform=target_transform)
    assert corruption in ["brightness",
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
    ## Check data unzipped:
    corruptiondir = os.path.join(root_dir, self.corruption)
    if not os.path.exists(corruptiondir):
      os.mkdir(corruptiondir)
    datapath = os.path.join(corruptiondir, "cifar10c_{}_data.npy".format(self.corruption))
    targetpath = os.path.join(corruptiondir, "cifar10c_{}_labels.npy".format(self.corruption))
    if not os.path.exists(datapath) or not os.path.exists(targetpath):
      zippath = os.path.join(here, "../../data/cifar10-c/{}.zip".format(self.corruption))
      assert os.path.exists(zippath);
      "Error: zipped cifar10-c dataset not located."

      unzip_to(zippath, corruptiondir)  ## creates cinic-10 dataset at requested location.

    ## Now get data and put it in memory:
    self.alldata = np.load(datapath)
    self.alltargets = list(np.load(targetpath))
    start_index = (self.level - 1) * 10000
    end_index = (self.level) * 10000
    self.data = self.alldata[start_index:end_index, :]
    self.targets = self.alltargets[start_index:end_index]
    ## Class-index mapping from original CIFAR10 dataset:
    self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    self.class_to_idx = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6,
                         'horse': 7, 'ship': 8, 'truck': 9}


class CINIC10(torchvision.datasets.ImageFolder):
  """Pytorch wrapper around the CINIC10 (imagenet only) dataset (https://github.com/BayesWatch/cinic-10). Assumes that the zipped dataset is already stored locally () using git LFS.

  """

  def __init__(self, root_dir, split, transform=None, target_transform=None, loader=None, is_valid_file=None,
               preload=False, labels=None):
    """
    :param root_dir: path to store data files.
    :param split: which split (train/test/val) of the data to grab.
    :param transform: an optional transform to be applied on PIL image samples.
    :param target_transform: an optional transform to be applied to targets.
    :param loader: an optional callable to load in images.
    :param is_valid_file: an optional loader to check image vailidity.
    :param preload: optionally preload all images into .data attribute
    :param labels: if labels are given, use these instead of class labels
    """
    assert str(root_dir).endswith("cinic-10"), "the directory must be called `cinic-10`"
    assert split in ["train", "test", "val"]
    self.labels = labels
    if not os.path.exists(os.path.join(root_dir, "train")):
      here = os.path.dirname(os.path.abspath(__file__))
      zippath = os.path.join(here, "../../data/cinic-10.zip")
      assert os.path.exists(zippath);
      "Error: zipped cinic-10 dataset not located."
      unzip_to(zippath, os.path.dirname(root_dir))  ## creates cinic-10 dataset at requested location.

    ## First, check if the dataset exists. If not, unzip it from local.
    args = {"transform": transform,
            "target_transform": target_transform,
            "loader": loader,
            "is_valid_file": is_valid_file}
    kws = {}
    for key in args:
      if args[key] is not None:
        kws[key] = args[key]
    super().__init__(os.path.join(root_dir, split), **kws)

    ## Class-index mapping from original CIFAR10 dataset:
    self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    self.class_to_idx = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6,
                         'horse': 7, 'ship': 8, 'truck': 9}
    if preload:
      self.data = np.stack([np.array(self.__getitem__(i)[0]) for i in range(len(self.samples))], axis=0)
      self.targets = [s[1] for s in self.samples]

  def __getitem__(self, idx):
    """
    optional override of getitem.
    """
    if self.labels is not None:
      img, _ = super().__getitem__(idx)
      label = self.labels[idx]
      return (img, label)
    else:
      return super().__getitem__(idx)


class CIFAR10_1(torchvision.datasets.vision.VisionDataset):
  """Pytorch wrapper around the CIFAR10.1 dataset (https://github.com/modestyachts/CIFAR-10.1)

  """

  def __init__(self, root_dir, version="v6", transform=None, train=False, target_transform=None):
    """
    :param root_dir: path to store data files.
    :param version: there is a v4 and v6 version of this data.
    :param transform: an optional transform to be applied on PIL image samples.
    :param target_transform: an optional transform to be applied to targets.
    **NB: there is an attribute "transforms" that we don't make use of in this class that the original CIFAR10 might.**
    """
    super().__init__(root_dir, transform=transform, target_transform=target_transform)
    # self.root = root_dir
    self.version = version
    # self.transform = transform
    self.train = train  ## should always be false.
    # self.target_transform = target_transform
    # self.transforms = None ## don't know what this parameter is..
    assert self.version in ["v4", "v6"]
    ## Download data
    self.datapath, self.targetpath = self.download()
    ## Now get data and put it in memory:
    self.data = np.load(self.datapath)
    self.targets = list(np.load(self.targetpath))
    ## Class-index mapping from original CIFAR10 dataset:
    self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    self.class_to_idx = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6,
                         'horse': 7, 'ship': 8, 'truck': 9}

  def download(self):
    """Download data from github.

    :returns: path where the data and label are located.
    """
    root_path = "https://raw.github.com/modestyachts/CIFAR-10.1/master/datasets/"
    data_name = "cifar10.1_{}_data.npy".format(self.version)
    label_name = "cifar10.1_{}_labels.npy".format(self.version)

    dataurl = os.path.join(root_path, data_name)
    datadestination = os.path.join(self.root, "data.npy")
    stream_download(dataurl, datadestination)
    labelurl = os.path.join(root_path, label_name)
    labeldestination = os.path.join(self.root, "labels.npy")
    stream_download(labelurl, labeldestination)
    return datadestination, labeldestination

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
    self.hparams = args
    self.mean = (0.47889522, 0.47227842, 0.43047404)
    self.std = (0.24205776, 0.23828046, 0.25874835)
    if args.get("custom_targets_eval_ood", False):
      self.set_targets_eval_ood = np.load(args.custom_targets_eval_ood)
    else:
      self.set_targets_eval_ood = None

  def train_dataloader(self):
    raise NotImplementedError("don't know the right transforms for this- will implement later")

  def val_dataloader(self):
    transform = T.Compose(
      [
        T.ToTensor(),
        T.Normalize(self.mean, self.std),
      ]
    )
    dataset = CINIC10(root_dir=os.path.join(self.hparams.data_dir, "cinic-10"), split="val", transform=transform)
    if self.set_targets_eval_ood is not None:
      dataset.targets = self.set_targets_eval_ood
      assert len(dataset.data) == len(dataset.targets), "number of examples, {} does not match targets {}".format(
        len(dataset.data), len(dataset.targets))
      assert dataset.data.shape[1] >= np.max(
        dataset.targets), "number of classes, {} does not match target index {}".format(dataset.data.shape[1],
                                                                                        np.max(dataset.targets))
    dataloader = DataLoader(
      dataset,
      batch_size=self.hparams.batch_size,
      num_workers=self.hparams.num_workers,
      drop_last=False,
      pin_memory=True,
    )
    return dataloader

  def test_dataloader(self):
    transform = T.Compose(
      [
        T.ToTensor(),
        T.Normalize(self.mean, self.std),
      ]
    )
    if self.set_targets_eval_ood is not None:
      dataset = CINIC10(root_dir=os.path.join(self.hparams.data_dir, "cinic-10"), split="test", transform=transform,
                        preload=True, labels=self.set_targets_eval_ood)
      dataset.targets = self.set_targets_eval_ood
      assert len(dataset.data) == len(dataset.targets), "number of examples, {} does not match targets {}".format(
        len(dataset.data), len(dataset.targets))
      assert dataset.data.shape[1] >= np.max(
        dataset.targets), "number of classes, {} does not match target index {}".format(dataset.data.shape[1],
                                                                                        np.max(dataset.targets))
    else:
      dataset = CINIC10(root_dir=os.path.join(self.hparams.data_dir, "cinic-10"), split="test", transform=transform,
                        preload=True)
    dataloader = DataLoader(
      dataset,
      batch_size=self.hparams.batch_size,
      num_workers=self.hparams.num_workers,
      drop_last=False,
      pin_memory=True,
    )
    return dataloader


class CIFAR10_1Data(pl.LightningDataModule):
  def __init__(self, args, version="v6"):
    super().__init__()
    self.hparams = args  ## check these.
    self.mean = (0.4914, 0.4822, 0.4465)
    self.std = (0.2471, 0.2435, 0.2616)
    self.version = version
    if args.get("custom_targets_eval_ood", False):
      self.set_targets_eval_ood = np.load(args.custom_targets_eval_ood)
    else:
      self.set_targets_eval_ood = None

  def train_dataloader(self):
    raise NotImplementedError

  def val_dataloader(self):
    transform = T.Compose(
      [
        T.ToTensor(),
        T.Normalize(self.mean, self.std),
      ]
    )
    dataset = CIFAR10_1(root_dir=self.hparams.data_dir, train=False, transform=transform, version=self.version)
    if self.set_targets_eval_ood is not None:
      dataset.targets = self.set_targets_eval_ood
      assert len(dataset.data) == len(dataset.targets), "number of examples, {} does not match targets {}".format(
        len(dataset.data), len(dataset.targets))
      assert dataset.data.shape[1] >= np.max(
        dataset.targets), "number of classes, {} does not match target index {}".format(dataset.data.shape[1],
                                                                                        np.max(dataset.targets))
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


class CIFAR10_CData(pl.LightningDataModule):
  """Dataset management- downloading, hyperparams, etc.

  :param args: the argparse parse_args output. key here is the root_dir parameter- we will work with a subdirectory of `root_dir` called cifar-10c .
  """

  def __init__(self, args):
    super().__init__()
    self.hparams = args
    self.mean = (0.4914, 0.4822, 0.4465)  ##? should we revise these?
    self.std = (0.2471, 0.2435, 0.2616)  ##?
    self.args = args
    if args.get("custom_targets_eval_ood", False):
      self.set_targets_eval_ood = np.load(args.custom_targets_eval_ood)
    else:
      self.set_targets_eval_ood = None

  def train_dataloader(self):
    raise NotImplementedError

  def val_dataloader(self):
    transform = T.Compose(
      [
        T.ToTensor(),
        T.Normalize(self.mean, self.std),
      ]
    )
    dataset = CIFAR10_C(root_dir=os.path.join(self.hparams.data_dir, "cifar10-c"), corruption=self.args.corruption,
                        level=self.args.level, transform=transform)
    if self.set_targets_eval_ood is not None:
      dataset.targets = self.set_targets_eval_ood
      assert len(dataset.data) == len(dataset.targets), "number of examples, {} does not match targets {}".format(
        len(dataset.data), len(dataset.targets))
      assert dataset.data.shape[1] >= np.max(
        dataset.targets), "number of classes, {} does not match target index {}".format(dataset.data.shape[1],
                                                                                        np.max(dataset.targets))
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


class CIFAR10Data(pl.LightningDataModule):
  def __init__(self, args):
    super().__init__()
    self.hparams = args
    self.mean = (0.4914, 0.4822, 0.4465)
    self.std = (0.2471, 0.2435, 0.2616)
    ## if softmax targets are given, parse.
    if args.get("custom_targets_train", False):
      # self.set_targets_train = parse_softmax(args.softmax_targets_train)
      ## training targets should be softmax! others should be binary.
      self.set_targets_train = np.load(args.custom_targets_train)
    else:
      self.set_targets_train = None
    if args.get("custom_targets_eval_ind", False):
      self.set_targets_eval_ind = np.load(args.custom_targets_eval_ind)
    else:
      self.set_targets_eval_ind = None

  def download_weights():
    url = (
      "https://rutgers.box.com/shared/static/gkw08ecs797j2et1ksmbg1w5t3idf5r5.zip"
    )

    # Streaming, so we can iterate over the response.
    r = requests.get(url, stream=True)

    # Total size in Mebibyte
    total_size = int(r.headers.get("content-length", 0))
    block_size = 2 ** 20  # Mebibyte
    t = tqdm(total=total_size, unit="MiB", unit_scale=True)

    with open("state_dicts.zip", "wb") as f:
      for data in r.iter_content(block_size):
        t.update(len(data))
        f.write(data)
    t.close()

    if total_size != 0 and t.n != total_size:
      raise Exception("Error, something went wrong")

    print("Download successful. Unzipping file...")
    path_to_zip_file = os.path.join(os.getcwd(), "state_dicts.zip")
    directory_to_extract_to = os.path.join(os.getcwd(), "cifar10_models")
    with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
      zip_ref.extractall(directory_to_extract_to)
      print("Unzip file successful!")

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
    dataset = CIFAR10(root=self.hparams.data_dir, train=True, transform=transform, download=True)
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

  def val_dataloader(self):
    transform = T.Compose(
      [
        T.ToTensor(),
        T.Normalize(self.mean, self.std),
      ]
    )
    dataset = CIFAR10(root=self.hparams.data_dir, train=False, transform=transform, download=True)
    if self.set_targets_eval_ind is not None:
      dataset.targets = self.set_targets_eval_ind
      assert len(dataset.data) == len(dataset.targets), "number of examples, {} does not match targets {}".format(
        len(dataset.data), len(dataset.targets))
      assert dataset.data.shape[1] >= np.max(
        dataset.targets), "number of classes, {} does not match target index {}".format(dataset.data.shape[1],
                                                                                        np.max(dataset.targets))
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


class CIFAR100Data(pl.LightningDataModule):
  def __init__(self, args):
    super().__init__()
    self.hparams = args
    self.mean = (0.4914, 0.4822, 0.4465)
    self.std = (0.2023, 0.1994, 0.2010)
    # hack to debug augmentation:
    self.train_aug = self.hparams.get('train_aug', True)

    ## if softmax targets are given, parse.
    if args.get("custom_targets_train", False):
      # self.set_targets_train = parse_softmax(args.softmax_targets_train)
      ## training targets should be softmax! others should be binary.
      self.set_targets_train = np.load(args.custom_targets_train)
    else:
      self.set_targets_train = None
    if args.get("custom_targets_eval_ind", False):
      self.set_targets_eval_ind = np.load(args.custom_targets_eval_ind)
    else:
      self.set_targets_eval_ind = None

  def train_dataloader(self, shuffle=True, aug=True):
    """added optional shuffle parameter for generating random labels.
    added optional aug parameter to apply augmentation or not.

    """
    if (aug is True) and (self.train_aug is True):
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
    dataset = CIFAR100(root=self.hparams.data_dir, train=True, transform=transform, download=True)
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

  def val_dataloader(self):
    transform = T.Compose(
      [
        T.ToTensor(),
        T.Normalize(self.mean, self.std),
      ]
    )
    dataset = CIFAR100(root=self.hparams.data_dir, train=False, transform=transform, download=True)
    if self.set_targets_eval_ind is not None:
      dataset.targets = self.set_targets_eval_ind
      assert len(dataset.data) == len(dataset.targets), "number of examples, {} does not match targets {}".format(
        len(dataset.data), len(dataset.targets))
      assert dataset.data.shape[1] >= np.max(
        dataset.targets), "number of classes, {} does not match target index {}".format(dataset.data.shape[1],
                                                                                        np.max(dataset.targets))
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


class CIFAR100Coarse(CIFAR100):
  """CIFAR100 with coarse labels.
  from https://github.com/ryanchankh/cifar100coarse/blob/master/cifar100coarse.py
  """

  def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
    super(CIFAR100Coarse, self).__init__(root, train, transform, target_transform, download)

    # update labels
    coarse_labels = np.array([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                              3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                              6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                              0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                              5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                              16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                              10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                              2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                              16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                              18, 1, 2, 15, 6, 0, 17, 8, 14, 13])
    self.targets = coarse_labels[self.targets]

    # update classes
    self.classes = [['beaver', 'dolphin', 'otter', 'seal', 'whale'],
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
                    ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
                    ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
                    ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']]


class CIFAR100CoarseData(pl.LightningDataModule):
  # TODO: inherit from CIFAR100Data
  def __init__(self, args):
    super().__init__()
    self.hparams = args
    self.mean = (0.4914, 0.4822, 0.4465)
    self.std = (0.2023, 0.1994, 0.2010)
    ## if softmax targets are given, parse.
    if args.get("custom_targets_train", False):
      # self.set_targets_train = parse_softmax(args.softmax_targets_train)
      ## training targets should be softmax! others should be binary.
      self.set_targets_train = np.load(args.custom_targets_train)
    else:
      self.set_targets_train = None
    if args.get("custom_targets_eval_ind", False):
      self.set_targets_eval_ind = np.load(args.custom_targets_eval_ind)
    else:
      self.set_targets_eval_ind = None

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
    dataset = CIFAR100Coarse(root=self.hparams.data_dir, train=True, transform=transform, download=True)
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

  def val_dataloader(self):
    transform = T.Compose(
      [
        T.ToTensor(),
        T.Normalize(self.mean, self.std),
      ]
    )
    dataset = CIFAR100Coarse(root=self.hparams.data_dir, train=False, transform=transform, download=True)
    if self.set_targets_eval_ind is not None:
      dataset.targets = self.set_targets_eval_ind
      assert len(dataset.data) == len(dataset.targets), "number of examples, {} does not match targets {}".format(
        len(dataset.data), len(dataset.targets))
      assert dataset.data.shape[1] >= np.max(
        dataset.targets), "number of classes, {} does not match target index {}".format(dataset.data.shape[1],
                                                                                        np.max(dataset.targets))
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