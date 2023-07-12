"""
Edited from
https://github.com/facebookresearch/classifier-balancing/blob/main/data/dataloader.py
"""
import os
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
import json
from tqdm import tqdm
from ..utils import BaseDataModule, stream_download


dir_path = os.path.dirname(os.path.realpath(__file__))



json2txt = {
  'train2018.json': 'iNaturalist18_train.txt',
  'val2018.json': 'iNaturalist18_val.txt'
}


def convert(json_file, txt_file):
  with open(json_file, 'r') as f:
    data = json.load(f)

  lines = []
  for i in tqdm(range(len(data['images']))):
    assert data['images'][i]['id'] == data['annotations'][i]['id']
    img_name = data['images'][i]['file_name']
    label = data['annotations'][i]['category_id']
    lines.append(img_name + ' ' + str(label) + '\n')

  with open(txt_file, 'w') as ftxt:
    ftxt.writelines(lines)


def prepare_data(root_dir):
  for k, v in json2txt.items():
    print('===> Converting {} to {}'.format(k, v), flush=True)
    srcfile = os.path.join(root_dir, k)
    convert(srcfile, v)


class iNaturalist18(ImageFolder):

  def __init__(self, root_dir, split, transform=None):
    self.img_path = []
    self.labels = []
    self.transform = transform

    assert split in ["train", "test", "val"]
    # check data exists
    if not os.path.exists(os.path.join(root_dir, "train")):

      datapath = os.path.join(root_dir, "train_val2018")
      if not os.path.exists(datapath):
        tarpath = datapath + ".tar.gz"
        if not os.path.exists(tarpath):
          self.download(root_dir)
        os.system("tar -zxvf %s -C %s"%(tarpath + "", root_dir))
        #os.system("rm %s"%(tarpath))
        prepare_data(root_dir)

    txt = './iNaturalist18_%s.txt'%(split)

    with open(txt) as f:
      for line in f:
        self.img_path.append(os.path.join(root_dir, line.split()[0]))
        self.labels.append(int(line.split()[1]))

  def download(self, root_dir):
    url = (
      "https://ml-inat-competition-datasets.s3.amazonaws.com/2018/train_val2018.tar.gz"
    )
    stream_download(url[0], os.path.join(root_dir, url[0].split('/')[-1]))

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, index):

    path = self.img_path[index]
    label = self.labels[index]

    with open(path, 'rb') as f:
      sample = Image.open(f).convert('RGB')

    if self.transform is not None:
      sample = self.transform(sample)

    return sample, label, index


class iNaturalist18Data(BaseDataModule):
  def __init__(self, args):
    super().__init__(args)
    self.mean = (0.466, 0.471, 0.380)
    self.std = (0.195, 0.194, 0.192)

    self.pin_memory = False

  def prepare_data(self):
      # download
      root_dir = os.path.join(self.hparams.data_dir, "iNaturalist18")
      iNaturalist18(root_dir, "train")

  def setup(self, stage=None):
      # Assign train/val dataset for use in dataloaders
      root_dir = os.path.join(self.hparams.data_dir, "iNaturalist18")
      train_set = iNaturalist18(root_dir, "train", transform=self.train_transform())
      valid_set = iNaturalist18(root_dir, "train", transform=self.valid_transform())
      test_set = iNaturalist18(root_dir, "val", transform=self.valid_transform())

      self._split_train_set(train_set, test_set, valid_set)

  def train_transform(self, aug=True):
    if aug is True:
      transform = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(self.mean, self.std),
      ])
      return transform
    else:
      return self.valid_transform()

  def valid_transform(self):
    transform = T.Compose([
      T.Resize(256),
      T.CenterCrop(224),
      T.ToTensor(),
      T.Normalize(self.mean, self.std),
    ])
    return transform
