import torch
from tqdm import tqdm
import numpy as np

def count_classes(data, num_classes=10, loader='train'):
  loaders = {
    'train': data.train_dataloader,
    'val': data.val_dataloader,
  }
  labels = torch.zeros(num_classes, dtype=torch.long)
  for idx, batch in tqdm(enumerate(loaders[loader]())):
    labels += torch.bincount(batch[1], minlength=num_classes)
  return labels


def split_dataset(dataset, split=0.9):
  shuffled_indices = np.random.permutation(len(dataset))
  train_idx = shuffled_indices[:int(split* len(dataset))]
  val_idx = shuffled_indices[int(split * len(dataset)):]
  return train_idx, val_idx


