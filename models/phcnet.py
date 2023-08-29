import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from modbayesian_torch.layers import LinearFlipout, LinearReparameterization
# logger = logging.getLogger(__name__)


class Encoder(nn.Module):
    def __init__(self, Lv, ks, dim = 2):
        super(Encoder, self).__init__()
        self.enc_block2d = nn.Sequential(
            nn.Conv2d(1, Lv[0], kernel_size=ks,stride=1,padding=math.ceil((ks-1)/2)),
            nn.BatchNorm2d(Lv[0]),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            # nn.Dropout(p=0.2),
            nn.Conv2d(Lv[0], Lv[1], kernel_size=ks,stride=1,padding=math.ceil((ks-1)/2)),
            nn.BatchNorm2d(Lv[1]),
            nn.ReLU(),
            nn.MaxPool2d(4,4),
            # nn.Dropout(p=0.2),
            nn.Conv2d(Lv[1], Lv[2], kernel_size=ks,stride=1,padding=math.ceil((ks-1)/2)),
            nn.BatchNorm2d(Lv[2]),
            nn.ReLU(),
            nn.MaxPool2d(4,4)
            )
        self.fcpart = nn.Sequential(
            nn.Linear(Lv[2] * 1 * 1, Lv[3]),
            nn.ReLU(),
            # nn.Dropout(p=0.2),
            nn.Linear(Lv[3], Lv[4]),
            )
        self.Lv = Lv
        self.dim = dim
        self.avgpool2d = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):

        x = self.enc_block2d(x)
        x = self.avgpool2d(x)
        x = x.view(-1, self.Lv[2] * 1 * 1)
        x = self.fcpart(x)
        return x

class SupCEWideResNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, Lv, ks=7, num_classes=5):
        super(SupCEWideResNet, self).__init__()
        self.encoder = Encoder(Lv, ks)
        self.fc = nn.Linear(Lv[4], num_classes)

    def forward(self, x):
        return self.fc(self.encoder(x))

class BayesCEWideResNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, Lv, ks=7, num_classes=5, prior_mu=0, prior_sigma=1, save_buffer_sd=False):
        super(BayesCEWideResNet, self).__init__()
        self.encoder = Encoder(Lv, ks)
        print(f"Using Reparam Bayes linear classifier {Lv[4]}, {num_classes}")
        self.fc = LinearReparameterization(in_features=Lv[4], out_features=num_classes,save_buffer_sd=save_buffer_sd,prior_mean=prior_mu,prior_variance=prior_sigma)

    def forward(self, x):
        return self.fc(self.encoder(x))
