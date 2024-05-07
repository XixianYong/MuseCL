from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import Inception3


class PlaceImageSkipGram(nn.Module):
    def __init__(self, embedding_dim):
        super(PlaceImageSkipGram, self).__init__()
        self.inception3 = Inception3_modified(aux_logits=False, transform_input=False)
        self.linear1 = nn.Linear(2048, embedding_dim)

    def forward(self, images):
        x1 = self.inception3(images)  # N x 2048
        result = self.linear1(x1)
        return result

    def load_CNN_params(self, CNN_model_path, device=torch.device('cpu')):
        old_params = torch.load(CNN_model_path, map_location=device)
        if CNN_model_path[-4:] == '.tar':  # The file is not a model state dict, but a checkpoint dict
            old_params = old_params['model_state_dict']
        del old_params['fc.weight']  # delete the unused parameters
        del old_params['fc.bias']  # delete the unused parameters
        self.inception3.load_state_dict(old_params, strict=False)
        print('Loaded pretrained CNN parameters from: ' + CNN_model_path)

    def only_train(self, trainable_params):
        for name, p in self.named_parameters():
            p.requires_grad = False
            for target in trainable_params:
                if target == name or target in name:
                    p.requires_grad = True
                    break


class Inception3_modified(Inception3):
    def forward(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 2048 x 1 x 1
        # x = F.dropout(x, training=self.training)
        x = F.dropout(x, training=False)
        # N x 2048 x 1 x 1
        x = x.view(x.size(0), -1)
        # N x 2048
        # x = self.fc(x)
        # # N x 1000 (num_classes)
        # if self.training and self.aux_logits:
        #     return _InceptionOuputs(x, aux)
        return x