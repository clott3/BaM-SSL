import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from modbayesian_torch.layers import LinearFlipout, LinearReparameterization
import torchvision

logger = logging.getLogger(__name__)


class SupCEResNet50(nn.Module):
    """encoder + classifier"""
    def __init__(self, num_classes=1000, dropout=0.0, non_linear=False, clas_depth=2):
        super(SupCEResNet50, self).__init__()
        self.encoder = torchvision.models.resnet50()
        self.encoder.fc = nn.Identity()
        self.dim_in = 2048

        if not non_linear:
            self.fc = nn.Linear(self.dim_in, num_classes)
        else:
            print("using MLP as Classifier with hidden dim ", self.dim_in)
            self.fc = nn.Sequential(
                    nn.Linear(self.dim_in, self.dim_in),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.dim_in, num_classes)
                )

    def forward(self, x):
        return self.fc(self.encoder(x))

class BayesCEResNet50(nn.Module):
    """encoder + classifier"""
    def __init__(self, num_classes=1000, non_linear=False, clas_depth=2, prior_mu=0, prior_sigma=1, flipout=False, reparam=False, save_buffer_sd=False):
        super(BayesCEResNet50, self).__init__()
        self.encoder = torchvision.models.resnet50()
        self.encoder.fc = nn.Identity()
        self.dim_in = 2048

        if not non_linear:
            if flipout:
                print(f"Using Flipout Bayes linear classifier {self.dim_in}, {num_classes}")
                self.fc = LinearFlipout(in_features=self.dim_in, out_features=num_classes,save_buffer_sd=save_buffer_sd,prior_mean=prior_mu,prior_variance=prior_sigma)
            elif reparam:
                print(f"Using Reparam Bayes linear classifier {self.dim_in}, {num_classes}")
                self.fc = LinearReparameterization(in_features=self.dim_in, out_features=num_classes,save_buffer_sd=save_buffer_sd,prior_mean=prior_mu,prior_variance=prior_sigma)
            else:
                raise "reparam or flipout"
                # print(f"Using BNN Bayes linear classifier {self.dim_in}, {num_classes}")
                # self.fc = bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=self.dim_in, out_features=num_classes,save_buffer_sd=save_buffer_sd)
        else:
            if flipout:
                raise
            else:
                print("using MLP as Bayes Classifier with hidden dim ", self.dim_in)
                self.fc = nn.Sequential(
                        bnn.BayesLinear(prior_mu=0, prior_sigma=1, in_features=self.dim_in, out_features=self.dim_in),
                        nn.ReLU(inplace=True),
                        bnn.BayesLinear(prior_mu=0, prior_sigma=1, in_features=self.dim_in, out_features=num_classes)
                        )

    def forward(self, x):
        return self.fc(self.encoder(x))
