import argparse
import copy
import csv
import os
import random
import time

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
import numpy as np


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def get_resnet18(pretrained, output_dim, ckpt_path):
    model = torchvision.models.resnet18(pretrained=pretrained)
    model.fc = Identity()
    checkpoint = torch.load(ckpt_path)
    checkpoint.pop('fc.weight')
    checkpoint.pop('fc.bias')
    model.load_state_dict(checkpoint)
    model.fc = torch.nn.Linear(512, output_dim)
    return model


transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor()])

model = get_resnet18(pretrained=True, output_dim=256, ckpt_path="")
model.load_state_dict(torch.load("", map_location=torch.device('cpu'))['model_state_dict'])
data_path = ""
image_list = np.load("").tolist()
image_list.sort()

emb = []

for image_name in tqdm(image_list):
    image_path = os.path.join(data_path, image_name + '.jpg')
    image = Image.open(image_path)
    image = transforms(image)
    image_emb = model(image.unsqueeze(0))
    emb.append(image_emb.detach().numpy())
    print(image_name)
    print(image_emb)

np.save('', np.array(emb))
print(emb)
