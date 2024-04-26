"""
python -m datatest_imbalanced.py
python -m unittest -k DatasetLoaderTest.test_imb_data_val datatest_imbalanced.py
"""
from pathlib import Path
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
from ml_collections import config_dict
import os
from pytorch_data.cifar.data_imbalanced import IMBALANCECIFAR10Data, IMBALANCECIFAR10DataAug, IMBALANCECIFAR10DataAug2, IMBALANCECIFAR10Data_v1
from pytorch_data.utils import count_classes
from pytorch_lightning import seed_everything
import torch

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
    cfg.batch_size = os.cpu_count()*16
    cfg.num_workers = os.cpu_count()
    return cfg

  @parameterized.named_parameters(
    ("cifar10_lt", "cifar10", IMBALANCECIFAR10Data, "exp", 0.01, 0.005),
    ("cifar10_lt_aug", "cifar10", IMBALANCECIFAR10DataAug, "exp", 0.01, 0.005),
    ("cifar10_lt_aug_v2", "cifar10", IMBALANCECIFAR10DataAug2, "exp", 0.01, 0.005),
  )
  def test_imb_data_loading(self, dataset_name, dataset_class, imb_type, imb_factor1, imb_factor2):
    # test data loading for imbalanced datasets
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

  @parameterized.named_parameters(
    ("cifar10_lt", "cifar10", IMBALANCECIFAR10Data_v1, "exp", 0.01, 0.1),
  )
  def test_imb_data_val(self, dataset_name, dataset_class, imb_type, imb_factor1, valid_size):
    # test validation data loader for imbalanced datasets
    num_classes = NUM_CLASSES[dataset_name]
    args = self.get_cfg()
    ind_data = dataset_class(args)
    ind_data.imb_type = imb_type
    ind_data.imb_factor = imb_factor1
    ind_data.valid_size = valid_size

    ind_data.prepare_data()
    ind_data.setup()
    labels_train = count_classes(ind_data, num_classes=num_classes, loader='train')
    num_val_samples = int(valid_size*labels_train.sum())

    for valid_type in ["random", "uniform", "match"]:
      ind_data.valid_type = valid_type
      ind_data.prepare_data()
      print('Finished preparing data', flush=True)
      ind_data.setup()
      print('Finished setting up data', flush=True)

      # get number of samples in validation set
      labels_val = count_classes(ind_data, num_classes=num_classes, loader='val')
      if valid_type == 'random':
        # if the validation type is random,
        # just check that the number of elements in the validation set should is 0:
        assert num_val_samples == labels_val.sum()
      elif valid_type == 'uniform':
        # if the validation type is uniform:
        # the number of elements in the validation set, there should be the same number of elements in each class
        assert labels_val.min() == labels_val.max()
        assert num_val_samples == labels_val.sum()
      elif valid_type == 'match':
        assert num_val_samples == labels_val.sum()
        assert (labels_train*valid_size - labels_val).abs().max() < num_classes
        # if the validation type is match:
        # the number of elements in the validation set should match those of the training set

  @parameterized.named_parameters(
  ("cifar10_lt", "cifar10", IMBALANCECIFAR10Data, "exp", 0.01),
  )
  def test_imb_data_seeded(self, dataset_name, dataset_class, imb_type, imb_factor):
    # test that changing the seed doesn't change the class fractions
    # but it changes the samples selected per class
    num_classes = NUM_CLASSES[dataset_name]
    args = self.get_cfg()
    ind_data = dataset_class(args)
    ind_data.imb_type = imb_type
    ind_data.imb_factor = imb_factor

    seeds = [0, 1, 2, 3]
    num_samples_p_class = np.zeros((len(seeds), num_classes))
    samples_per_seed = []
    for seed in seeds:
      ind_data.seed = seed
      ind_data.prepare_data()
      print('Finished preparing data', flush=True)
      ind_data.setup()
      num_samples_p_class[seed] = np.unique(ind_data.train_dataset.targets, return_counts=True)[1]
      samples_per_seed.append(np.sort(ind_data.train_dataset.imbalanced_train_indices))

    # check that regardless of the seed, the sample number of samples per class are selected
    assert np.alltrue([np.alltrue(num_samples_p_class[0] == num_samples_p_class[j]) for j in range(1, len(seeds))])

    # check that different samples are selected given different seeds
    assert np.alltrue([~np.alltrue(samples_per_seed[0] == samples_per_seed[j]) for j in range(1, len(seeds))])

  @parameterized.named_parameters(
  ("only_all_seed", "only_all_seed"),
  ("data_seed_diff", "data_seed_diff"),
  ("data_seed_same", "data_seed_same"),
  ("batch_seed_diff", "batch_seed_diff"),
  )
  def test_compare_seeds(self, seed_type):
    """
    Seed options for two dataloaders
    value | all_seed | data_seed | seed_batch | output
    --------------------------------------
     only_all_seed    | equal    | -         | - | two identical datasets
     data_seed_diff   | equal    | diff      | - | two diff datasets
     data_seed_same   | diff    | equal      | - | two identical datasets
     batch_seed_diff  | diff    | equal      | diff | two identical datasets in different order
    """
    args1 = self.get_cfg()
    args2 = self.get_cfg()

    if seed_type in ("only_all_seed", "data_seed_same", ):
      args1.seed_all = 0
      args2.seed_all = 0

      if seed_type in ("data_seed_same"):
        args1.seed = 0
        args2.seed = 0

      train_comparison = lambda x, y: np.alltrue(x == y)
      val_comparison = train_comparison
      batch_comparison = lambda x, y: torch.all(x == y)

    elif seed_type in ("data_seed_diff"):
      args1.seed_all = 0
      args2.seed_all = 0

      args1.seed = 0
      args2.seed = 1

      train_comparison = lambda x, y: ~np.alltrue(x == y)
      val_comparison = train_comparison
      batch_comparison = lambda x, y: ~torch.all(x == y)

    elif seed_type in ("batch_seed_diff"):
      args1.seed_all = 0
      args2.seed_all = 0

      args1.seed = 0
      args2.seed = 0

      args1.seed_batch = 0
      args2.seed_batch = 1

      train_comparison = lambda x, y: np.alltrue(x == y)
      val_comparison = train_comparison
      batch_comparison = lambda x, y: ~torch.all(x == y)

    # Now make sure these hold
    train_samples_per_seed = []
    val_samples_per_seed = []
    batch_sample_order = []

    for args in [args1, args2]:
      if args.get("seed_all", None) is not None:
        seed_everything(args.seed_all)

      ind_data = IMBALANCECIFAR10Data(args)
      ind_data.prepare_data()
      ind_data.setup()
      train_samples_per_seed.append(np.sort(ind_data.train_dataset.imbalanced_train_indices))
      val_samples_per_seed.append(np.sort(ind_data.train_dataset.imbalanced_train_indices))
      batch_sample_order.append(next(iter(ind_data.train_dataloader()))[1])

    assert train_comparison(*train_samples_per_seed)
    assert val_comparison(*val_samples_per_seed)
    assert batch_comparison(*batch_sample_order)


if __name__ == '__main__':
  absltest.main()
