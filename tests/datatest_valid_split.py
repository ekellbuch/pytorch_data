from absl.testing import absltest
from ml_collections import config_dict
from pathlib import Path
from absl.testing import parameterized
from pytorch_data.cifar.data import CIFAR10Data, CIFAR100Data
from pytorch_data.cifar.data_imbalanced import IMBALANCECIFAR10Data, IMBALANCECIFAR100Data
DATA_DIR = Path.home() / "pytorch_datasets"
import numpy as np

# val set: # samples per class
BASELINE_VAL_P_CLASS = {
  "cifar10": 1000,
  "cifar10_1": 200,
  "cinic10": 7000,
  "cifar100": 100,
  "tiny_imagenet": 50,
}


BASELINE_VAL = lambda x: BASELINE_VAL_P_CLASS[x] * NUM_CLASSES[x] if x != "inaturalist18" else 24426
BASELINE_TRAIN = lambda x: BASELINE_TRAIN_P_CLASS[x] * NUM_CLASSES[x] if x != "inaturalist18" else 437513


# train set: # samples per class
BASELINE_TRAIN_P_CLASS = {
  "cifar10": 5000,
  "cifar100": 500,
  "tiny_imagenet": 500,
}

NUM_CLASSES = {
  "cifar10": 10,
  "cifar10_1": 10,
  "cinic10": 10,
  "cifar100": 100,
  "cifar100coarse": 20,
  "tiny_imagenet": 200,
}


class DatasetLoaderTest(parameterized.TestCase):
  def get_cfg(self):
    """Load all toy data config file without hydra."""
    cfg = config_dict.ConfigDict()
    cfg.data_dir = DATA_DIR
    cfg.batch_size = 128
    cfg.num_workers = 8
    cfg.seed = 0
    return cfg

  @parameterized.named_parameters(
    ("cifar10_no_split", "cifar10", 0, CIFAR10Data),
    ("cifar10_1split", "cifar10", 0.1, CIFAR10Data),
    ("cifar10_5split", "cifar10", 0.5, CIFAR10Data),
    ("cifar100_1split", "cifar10", 0.1, CIFAR100Data),
    ("cifar100_5split", "cifar10", 0.5, CIFAR100Data),
  )
  def test_data_loading(self, dataset_name, valid_size, dataset_class):
    num_classes = NUM_CLASSES[dataset_name]
    args = self.get_cfg()
    args.valid_size = valid_size
    ind_data = dataset_class(args)
    ind_data.prepare_data()
    ind_data.setup()
    # split train into train and valid set
    if valid_size == 0:
      self.assertEqual((len(ind_data.test_dataset), BASELINE_VAL(dataset_name)))
    else:
      self.assertEqual(len(ind_data.val_dataset), np.floor(len(ind_data.train_dataset)*valid_size))

  @parameterized.named_parameters(
    ("cifar10lt_no_split", "cifar10", 0, IMBALANCECIFAR10Data),
    ("cifar10lt_1split", "cifar10", 0.1, IMBALANCECIFAR10Data),
    ("cifar10lt_5split", "cifar10", 0.5, IMBALANCECIFAR10Data),
    ("cifar100lt_1split", "cifar10", 0.1, IMBALANCECIFAR100Data),
    ("cifar100lt_5split", "cifar10", 0.5, IMBALANCECIFAR100Data),
  )
  def test_data_loading(self, dataset_name, valid_size, dataset_class):
    num_classes = NUM_CLASSES[dataset_name]
    args = self.get_cfg()
    args.valid_size = valid_size
    ind_data = dataset_class(args)
    ind_data.prepare_data()
    ind_data.setup()
    # split train into train and valid set
    if valid_size == 0:
      self.assertEqual(len(ind_data.test_dataset), BASELINE_VAL(dataset_name))
    else:
      self.assertEqual(len(ind_data.val_dataset), np.floor(len(ind_data.train_dataset)*valid_size))


if __name__ == '__main__':
  absltest.main()
