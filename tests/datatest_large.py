from pathlib import Path

import torch
from absl.testing import absltest
from absl.testing import parameterized
from ml_collections import config_dict

from pytorch_data.inaturalist.data import iNaturalistData
from pytorch_data.inaturalist.data_inat18 import iNaturalist18Data
from pytorch_data.utils import count_classes

DATA_DIR = Path.home() / "pytorch_datasets"

# val set: # samples per class
BASELINE_VAL_P_CLASS = {
  "imagenet": 50,
}


BASELINE_VAL = lambda x: BASELINE_VAL_P_CLASS[x] * NUM_CLASSES[x] if x != "inaturalist18" else 24426
BASELINE_TRAIN = lambda x: BASELINE_TRAIN_P_CLASS[x] * NUM_CLASSES[x] if x != "inaturalist18" else 437513


# train set: # samples per class
BASELINE_TRAIN_P_CLASS = {
  "tiny_imagenet": 500,
}

NUM_CLASSES = {
  "imagenet": 1000,
  "inaturalist18": 8142
}


class DatasetLoaderTest(parameterized.TestCase):
  def get_cfg(self):
    """Load all toy data config file without hydra."""
    cfg = config_dict.ConfigDict()
    cfg.data_dir = DATA_DIR
    cfg.batch_size = 128
    cfg.num_workers = torch.get_num_threads()
    return cfg

  @parameterized.named_parameters(
    ("inaturalist18", "inaturalist18", iNaturalistData),
    #("imagenet", "imagenet", ImageNetData),
  )
  def test_data_loading(self, dataset_name, dataset_class):
    num_classes = NUM_CLASSES[dataset_name]
    args = self.get_cfg()
    ind_data = dataset_class(args)
    ind_data.prepare_data()
    ind_data.setup()
    labels = count_classes(ind_data, num_classes=num_classes, loader='val').sum()
    self.assertEqual(labels, BASELINE_VAL(dataset_name))

  @parameterized.named_parameters(
    ("inaturalist18_cmp", "inaturalist18"),
    #("imagenet", "imagenet", ImageNetData),
  )
  def test_data_loading(self, dataset_name):
    num_classes = NUM_CLASSES[dataset_name]
    args = self.get_cfg()
    ind_data = iNaturalistData(args)
    ind_data.prepare_data()
    ind_data.setup()
    labels = count_classes(ind_data, num_classes=num_classes, loader='val').sum()

    ind_data = iNaturalist18Data(args)
    ind_data.prepare_data()
    ind_data.setup()
    labels2 = count_classes(ind_data, num_classes=num_classes, loader='val').sum()

    self.assertEqual(labels, labels2)


if __name__ == '__main__':
  absltest.main()
