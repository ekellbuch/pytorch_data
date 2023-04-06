import os
import numpy as np
import torch
import pandas as pd
from typing import Any, Iterable, Optional, Tuple, Union
import sklearn.datasets as datasets
from scipy.io import loadmat
import logging


def _get_dataset_numpy(dataset: str, data_dir: str):
  if dataset == "california":
    # california 20640 8
    _Xs, _ys = datasets.fetch_california_housing(data_home=data_dir, download_if_missing=True, return_X_y=True)
  elif dataset == "diabetes":
    # diabetes 442 10
    diabetes = datasets.load_diabetes()  # load data
    _Xs, _ys = diabetes.data, diabetes.target
  elif dataset == "digits":
    # digits 1797 64
    _Xs, _ys = datasets.load_digits(return_X_y=True)
    _ys = _ys % 2
  elif dataset in ["concrete", "energy", "pol", "wine"]:
    # concrete 1030 8
    # energy 768 8
    # pol 15000 26
    # wine 1599 11
    _data = loadmat(os.path.join(data_dir, f"{dataset}.mat"))["data"]
    _Xs = _data[..., :-1]
    _ys = _data[..., -1]
  elif dataset == "howel":
    # howel 352 x 1
    # https://github.com/probml/pyprobml/blob/master/notebooks/book2/15/linreg_height_weight.ipynb
    url = "https://raw.githubusercontent.com/fehiepsi/rethinking-numpyro/master/data/Howell1.csv"
    Howell1 = pd.read_csv(url, sep=";")
    d = Howell1
    d.info()
    d.head()
    # get data for adults
    d2 = d[d.age >= 18]
    _Xs = d2.height.values[:, None]
    _ys = d2.weight.values
  else:
    raise ValueError(dataset)
  return _Xs, _ys


class PreprocessTransform(object):
  def __init__(self,
               remove_feature_no_std=True,
               center_response=True):
    self.remove_feature_no_std = remove_feature_no_std
    self.center_response = center_response

  def run_steps(self):

    # remove features with no std
    if self.remove_feature_no_std:
      self.fn_remove_feature_no_std()

    # center inputs
    self.fn_center_inputs()

    # center responses
    if self.center_response:
      self.fn_center_responses()

  def assign_inputs(self,
                    X: torch.Tensor,
                    X_test: torch.Tensor,
                    y: torch.Tensor,
                    y_test: torch.Tensor):
    self.X = X
    self.X_test = X_test
    self.y = y
    self.y_test = y_test

  def fn_remove_feature_no_std(self):
    # Remove features with no std
    _no_std = self.X.std(dim=-2).min(dim=0)[0].nonzero().squeeze(-1)
    self.X = self.X[..., _no_std]
    self.X_test = self.X_test[..., _no_std]

  def fn_center_inputs(self):
    # Center inputs
    self.x_mean = self.X.mean(dim=-2, keepdim=True)
    self.x_std = self.X.std(dim=-2, keepdim=True)
    self.X = self.X.sub(self.x_mean).div(self.x_std)
    self.X_test = self.X_test.sub(self.x_mean).div(self.x_std)

  def fn_center_responses(self):
    # Optionally center responses
    self.y_mean = self.y.mean(dim=-1, keepdim=True)
    self.y_std = self.y.std(dim=-1, keepdim=True)
    self.y = self.y.sub(self.y_mean).div(self.y_std)
    self.y_test = self.y_test.sub(self.y_mean).div(self.y_std)


class BaselineDataset(object):
  def __init__(self, dataset: str,
               N: int,
               N_test: int = 0,
               num_trials: int = 1,
               seed: Optional[int] = None,
               device: str = 'cpu',
               dtype: Any = torch.double,
               verbose: bool = False,
               data_dir: Optional[str] = "datasets",
               **kwargs):
    self.dataset = dataset
    self.N = N
    self.N_test = N_test
    self.num_trials = num_trials
    self.seed = seed
    self.device = device
    self.dtype = dtype
    self.data_dir=data_dir
    self.constructed = False
    del kwargs

    # Logging?
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    if seed is not None:
      np.random.seed(seed)
      torch.manual_seed(seed)

  def get_raw_data(self):
    _Xs, _ys = _get_dataset_numpy(self.dataset, self.data_dir)

    self._Xs = _Xs
    self._ys = _ys
    self.N_tot = _Xs.shape[-2]
    self.D_x = _Xs.shape[-1]

  def set_Ntest(self):
    # Get data
    N_test = min(self.N_tot - self.N, self.N_test) if self.N_test else self.N_tot - self.N

    if N_test != self.N_test:
      self.N_test = N_test
      logging.info(f"Setting N_test to {N_test}.")

  def get_perms(self):
    # Construct multiple trials - multiple dataset splits and frequencies
    data_perms = torch.stack([torch.randperm(self.N_tot, device=self.device) for _ in range(self.num_trials)], dim=-2)
    self.set_Ntest()
    N_extra = self.N_tot - self.N_test - self.N
    if N_extra:
      train_indices, test_indices, _ = data_perms.split([self.N, self.N_test, N_extra], dim=-1)
    else:
      train_indices, test_indices = data_perms.split([self.N, self.N_test], dim=-1)

    self.train_indices = train_indices
    self.test_indices = test_indices

  def split_data(self):
    # get_perm data
    self.get_perms()

    # Construct datasets
    Xs = (self._Xs if torch.is_tensor(self._Xs) else torch.tensor(self._Xs)).to(dtype=self.dtype, device=self.device)
    ys = (self._ys if torch.is_tensor(self._ys) else torch.tensor(self._ys)).to(dtype=self.dtype, device=self.device)

    self.X = Xs[self.train_indices]
    self.y = ys[self.train_indices]
    self.X_test = Xs[self.test_indices]
    self.y_test = ys[self.test_indices]

  def remove_feature_no_std(self):
    # Remove features with no std
    D_origin = self.X.size(-1)
    _no_std = self.X.std(dim=-2).min(dim=0)[0].nonzero().squeeze(-1)
    self.X = self.X[..., _no_std]
    self.D_x = self.X.size(-1)
    self.X_test = self.X_test[..., _no_std]

    if D_origin > self.D_x:
      logging.info(f"Removed {D_origin- self.D_x}/{D_origin} featurs with no std.")

  def center_inputs(self):
    # Center inputs
    self.x_mean = self.X.mean(dim=-2, keepdim=True)
    self.x_std = self.X.std(dim=-2, keepdim=True)
    self.X = self.X.sub(self.x_mean).div(self.x_std)
    self.X_test = self.X_test.sub(self.x_mean).div(self.x_std)

    logging.info(f"X mean/std: {self.X.mean().item()}, {self.X.std(dim=-2).mean().item()}")
    logging.info(f"X_test mean/std: {self.X_test.mean().item()}, {self.X_test.std(dim=-2).mean().item()}")

  def center_responses(self):
    # Optionally center responses
    self.y_mean = self.y.mean(dim=-1, keepdim=True)
    self.y_std = self.y.std(dim=-1, keepdim=True)
    self.y = self.y.sub(self.y_mean).div(self.y_std)
    self.y_test = self.y_test.sub(self.y_mean).div(self.y_std)

  def _get_dataset(self):

    # assign raw data
    self.get_raw_data()

    # construct datasets
    self.split_data()

    # preprocess dataset
    pre_data = PreprocessTransform()
    pre_data.assign_inputs(self.X, self.X_test,
                           self.y, self.y_test)
    pre_data.run_steps()

    self.constructed = True

  def make_label_one_hot(self):
    # Make labels one-hot
    self.y = torch.nn.functional.one_hot(self.y)
    self.y_test = torch.nn.functional.one_hot(self.y_test)

  def get_as_numpy(self):
    self.get_dataset()
    X = self.X.squeeze(0).cpu().numpy()
    y = self.y.squeeze(0).cpu().numpy()
    X_test = self.X_test.squeeze(0).cpu().numpy()
    y_test = self.y_test.squeeze(0).cpu().numpy()
    return X, y, X_test, y_test

  def get_dataset(self):
    if not self.constructed:
      self._get_dataset()
