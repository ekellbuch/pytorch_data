import torch
from tqdm import tqdm
import numpy as np

def count_classes(data, num_classes=10):
  labels = torch.zeros(num_classes, dtype=torch.long)
  for idx, batch in tqdm(enumerate(data.train_dataloader(shuffle=False, aug=False))):
    labels += torch.bincount(batch[1], minlength=num_classes)
  return labels


def split_dataset(dataset, split=0.8):
  shuffled_indices = np.random.permutation(len(dataset))
  train_idx = shuffled_indices[:int(split* len(dataset))]
  val_idx = shuffled_indices[int(split * len(dataset)):]
  return train_idx, val_idx


