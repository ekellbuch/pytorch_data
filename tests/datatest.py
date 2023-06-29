
from absl.testing import absltest
from ml_collections import config_dict
from pathlib import Path
from absl.testing import parameterized
from pytorch_data.utils import count_classes
from pytorch_data.cifar.data import CIFAR10Data, CIFAR100Data
from pytorch_data.imagenet.data import ImageNetData
from pytorch_data.cifar.data_imbalanced import IMBALANCECIFAR10Data, IMBALANCECIFAR100Data
DATA_DIR = Path.home() / "pytorch_datasets"


# val set sample size per class
BASELINE_VAL = {
  "cifar10": 1000,
  "cifar100": 100,
  "imagenet": 50,
}

BASELINE_TRAIN = {
  "cifar10": 6000,
  "cifar100": 500,
}

class DatasetLoaderTest(parameterized.TestCase):
    def get_cfg(self):
        """Load all toy data config file without hydra."""
        cfg = config_dict.ConfigDict()
        cfg.data_dir = DATA_DIR
        cfg.batch_size = 128
        cfg.num_workers = 8
        cfg.train_shuffle = False
        return cfg
    @parameterized.named_parameters(
       ("cifar10", "cifar10", CIFAR10Data, 10),
       ("cifar100","cifar100", CIFAR100Data, 100),
       ("imagenet", "imagenet", ImageNetData, 1000),
    )
    def test_data_loading(self, dataset_name, dataset_class, num_classes):

        args = self.get_cfg()
        ind_data = dataset_class(args)

        labels = count_classes(ind_data, num_classes=num_classes, loader='val').sum()
        self.assertEqual(labels, BASELINE_VAL[dataset_name]*num_classes)
    @parameterized.named_parameters(
        ("imbalancecifar10", "cifar10", IMBALANCECIFAR10Data, 10, "exp", 0.01),
        ("imbalancecifar100", "cifar100", IMBALANCECIFAR100Data, 100, "exp", 0.01),
    )
    def test_imb_data_loading(self, dataset_name, dataset_class, num_classes, imb_type, imb_factor):
        args = self.get_cfg()
        ind_data = dataset_class(args)
        ind_data.imb_type = imb_type
        ind_data.imb_factor = imb_factor

        # train set should be imbalanced
        labels = count_classes(ind_data, num_classes=num_classes, loader='train').sum()
        self.assertGreater(BASELINE_TRAIN[dataset_name]*num_classes, labels)
        # val set should be the same
        labels = count_classes(ind_data, num_classes=num_classes, loader='val').sum()
        self.assertEqual(labels, BASELINE_VAL[dataset_name]*num_classes)


if __name__ == '__main__':
  absltest.main()