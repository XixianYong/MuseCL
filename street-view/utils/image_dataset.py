from __future__ import division
from __future__ import print_function

import os

import pandas as pd
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class PlaceImagePairDataset(Dataset):
    def __init__(self, root_dir, path_list, transform):
        self.root_dir = root_dir
        self.path_list = path_list
        self.transform = transform

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        anc_path, pos_path, neg_path = self.path_list[idx]

        anc_path1 = os.path.join(self.root_dir, anc_path)
        anc_image = Image.open(anc_path1)

        pos_path1 = os.path.join(self.root_dir, pos_path)
        pos_image = Image.open(pos_path1)

        neg_path1 = os.path.join(self.root_dir, neg_path)
        neg_image = Image.open(neg_path1)

        if not pos_image.mode == 'RGB':
            pos_image = pos_image.convert('RGB')
        if not anc_image.mode == 'RGB':
            anc_image = anc_image.convert('RGB')
        if not neg_image.mode == 'RGB':
            neg_image = neg_image.convert('RGB')
        anc_image = self.transform(anc_image) # 797*299
        pos_image = self.transform(pos_image)
        neg_image = self.transform(neg_image)
        sample = [anc_image, pos_image, neg_image]
        return sample

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 4 else pad_sequence(dat[i], True) if i < 6 else dat[i].tolist() for i in
                dat]


class ImageDataset(Dataset):
    def __init__(self, root_dir, path_list, transform):
        self.root_dir = root_dir
        self.path_list = path_list
        self.transform = transform

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        # anc_path, fips = self.path_list[idx]
        fips, anc_path = self.path_list[idx]
        anc_path1 = os.path.join(self.root_dir, anc_path)
        anc_image = Image.open(anc_path1)
        if not anc_image.mode == 'RGB':
            anc_image = anc_image.convert('RGB')
        anc_image = self.transform(anc_image)
        sample = [anc_image, fips]
        return sample
