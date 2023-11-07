from pathlib import Path

from absl.testing import absltest
from absl.testing import parameterized
from ml_collections import config_dict

from pytorch_data.cifar.data import CIFAR10Data, CIFAR100Data, CIFAR10_1Data, CINIC10_Data
from pytorch_data.cifar.data_imbalanced import IMBALANCECIFAR10Data, IMBALANCECIFAR10DataAug, IMBALANCECIFAR10DataAug_v2
from pytorch_data.imagenet.data import TinyImagenetData
from pytorch_data.utils import count_classes

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
    ("cifar10_lt", "cifar10", IMBALANCECIFAR10Data, "exp", 0.01, 0.005),
    ("cifar10_lt_aug", "cifar10", IMBALANCECIFAR10DataAug, "exp", 0.01, 0.005),
    ("cifar10_lt_aug_v2", "cifar10", IMBALANCECIFAR10DataAug_v2, "exp", 0.01, 0.005),
  )
  def test_imb_data_loading(self, dataset_name, dataset_class, imb_type, imb_factor1, imb_factor2):
    num_classes = NUM_CLASSES[dataset_name]
    args = self.get_cfg()
    ind_data = dataset_class(args)
    ind_data.imb_type = imb_type
    ind_data.imb_factor = imb_factor1
    ind_data.prepare_data()
    print('Finished preparing data', flush=True)
    ind_data.setup()
    print('Finished setting up data', flush=True)
    # train set should be imbalanced
    labels1 = count_classes(ind_data, num_classes=num_classes, loader='train').sum()
    print(imb_factor1, count_classes(ind_data, num_classes=num_classes, loader='train'))

    args = self.get_cfg()
    ind_data = dataset_class(args)
    ind_data.imb_type = imb_type
    ind_data.imb_factor = imb_factor2
    ind_data.prepare_data()
    ind_data.setup()
    # train set should be imbalanced
    labels2 = count_classes(ind_data, num_classes=num_classes, loader='train').sum()
    print(imb_factor2, count_classes(ind_data, num_classes=num_classes, loader='train'))
    self.assertGreater(labels1, labels2)


if __name__ == '__main__':
  absltest.main()
