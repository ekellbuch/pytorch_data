from pathlib import Path

from absl.testing import absltest
from absl.testing import parameterized
from ml_collections import config_dict

from pytorch_data.cifar.data import CIFAR10Data, CIFAR100Data, CIFAR10_1Data, CINIC10_Data
from pytorch_data.cifar.data_imbalanced import IMBALANCECIFAR10Data, IMBALANCECIFAR100Data, IMBALANCECIFAR10DataAug, IMBALANCECIFAR100DataAug
from pytorch_data.imagenet.data import TinyImagenetData

DATA_DIR = Path.home() / "pytorch_datasets"

# val set: # samples per class
BASELINE_VAL_P_CLASS = {
  "cifar10": 1000,
  "cifar10_1": 200,
  "cinic10": 7000,
  "cifar100": 100,
  "tiny_imagenet": 50,
}


BASELINE_VAL = lambda x: BASELINE_VAL_P_CLASS[x] * NUM_CLASSES[x]
BASELINE_TRAIN = lambda x: BASELINE_TRAIN_P_CLASS[x] * NUM_CLASSES[x]


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
    return cfg

  @parameterized.named_parameters(
    ("cifar10", "cifar10", CIFAR10Data),
    ("cinic10", "cinic10", CINIC10_Data),
    ("cifar10_1", "cifar10_1", CIFAR10_1Data),
    ("cifar100", "cifar100", CIFAR100Data),
    ("tiny_imagenet", "tiny_imagenet", TinyImagenetData),
  )
  def test_data_loading(self, dataset_name, dataset_class):
    args = self.get_cfg()
    ind_data = dataset_class(args)
    ind_data.prepare_data()
    ind_data.setup()
    self.assertEqual(len(ind_data.val_dataset), BASELINE_VAL(dataset_name))

  @parameterized.named_parameters(
    ("cifar10_lt", "cifar10", IMBALANCECIFAR10Data, "exp", 0.01),
    ("cifar100_lt", "cifar100", IMBALANCECIFAR100Data, "exp", 0.01),
    ("cifar10_lt_aug", "cifar10", IMBALANCECIFAR10DataAug, "exp", 0.01),
    ("cifar100_lt_aug", "cifar100", IMBALANCECIFAR100DataAug, "exp", 0.01),
  )
  def test_imb_data_loading(self, dataset_name, dataset_class, imb_type, imb_factor):
    args = self.get_cfg()
    ind_data = dataset_class(args)
    ind_data.imb_type = imb_type
    ind_data.imb_factor = imb_factor
    ind_data.prepare_data()
    ind_data.setup()
    # train set should be imbalanced
    self.assertGreater(BASELINE_TRAIN(dataset_name), len(ind_data.train_dataset))
    # val set should be the same
    self.assertEqual(len(ind_data.val_dataset), BASELINE_VAL(dataset_name))


if __name__ == '__main__':
  absltest.main()
