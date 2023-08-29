import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from modbayesian_torch.layers import LinearFlipout, LinearReparameterization

logger = logging.getLogger(__name__)


def mish(x):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function (https://arxiv.org/abs/1908.08681)"""
    return x * torch.tanh(F.softplus(x))


class PSBatchNorm2d(nn.BatchNorm2d):
    """How Does BN Increase Collapsed Neural Network Filters? (https://arxiv.org/abs/2001.11216)"""

    def __init__(self, num_features, alpha=0.1, eps=1e-05, momentum=0.001, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha

    def forward(self, x):
        return super().forward(x) + self.alpha


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.drop_rate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, drop_rate, activate_before_residual))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, num_classes, depth=28, widen_factor=2, drop_rate=0.0,
                head='mlp', head_depth=2, hidden_dim=512, feat_dim=512, in_channels=3):
        super(WideResNet, self).__init__()
        channels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(in_channels, channels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(
            n, channels[0], channels[1], block, 1, drop_rate, activate_before_residual=True)
        # 2nd block
        self.block2 = NetworkBlock(
            n, channels[1], channels[2], block, 2, drop_rate)
        # 3rd block
        self.block3 = NetworkBlock(
            n, channels[2], channels[3], block, 2, drop_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(channels[3], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # if not non_linear:
        #     self.fc = nn.Linear(channels[3], num_classes)
        # else:
        #     list_layers = [nn.Linear(channels[3],channels[3]), nn.LeakyReLU(negative_slope=0.1, inplace=True)]
        #     for _ in range(depth-2):
        #         list_layers += [nn.Linear(channels[3],channels[3]), nn.LeakyReLU(negative_slope=0.1, inplace=True)]
        #     list_layers += [nn.Linear(channels[3],num_classes)]
        #     print("using MLP as Classifier with hidden dim={} and depth={}".format(channels[3],depth))
        #     self.fc = nn.Sequential(*list_layers)

        self.channels = channels[3]



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.channels)
        return out


def build_wideresnet(depth, widen_factor, dropout, num_classes, in_channels):
    logger.info(f"Model: WideResNet {depth}x{widen_factor}")
    return WideResNet(depth=depth,
                      widen_factor=widen_factor,
                      drop_rate=dropout,
                      num_classes=num_classes,
                      in_channels=in_channels)


class SupCEWideResNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, in_channels=3,num_classes=10, depth=28, widen_factor=2, dropout=0.0, non_linear=False, clas_depth=2):
        super(SupCEWideResNet, self).__init__()
        self.encoder = build_wideresnet(depth=depth,
                          widen_factor=widen_factor,
                          dropout=dropout,
                          num_classes=num_classes,
                          in_channels=in_channels)
        self.dim_in = self.encoder.channels

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

class BayesCEWideResNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, in_channels=3, num_classes=10, depth=28, widen_factor=2, dropout=0.0, non_linear=False, clas_depth=2, prior_mu=0, prior_sigma=1, flipout=False, reparam=False, save_buffer_sd=False):
        super(BayesCEWideResNet, self).__init__()
        self.encoder = build_wideresnet(depth=depth,
                          widen_factor=widen_factor,
                          dropout=dropout,
                          num_classes=num_classes,
                          in_channels=in_channels)
        self.dim_in = self.encoder.channels

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

class ProjectionMLP(nn.Module):
    def __init__(self, dim_in, hidden_dim=512, feat_dim=512, proj_depth=2):
        super().__init__()
        print("using MLP head with depth={}, in_dim={}, hidden_dim={}, feat_dim={}".format(proj_depth,dim_in,hidden_dim,feat_dim))
        if proj_depth == 1:
            list_layers = [nn.Linear(dim_in,feat_dim)]
        else:
            list_layers = [nn.Linear(dim_in, hidden_dim),
                       nn.BatchNorm1d(hidden_dim),
                       nn.ReLU(inplace=True)]
            for _ in range(proj_depth):
                list_layers += [nn.Linear(hidden_dim, hidden_dim),
                           nn.BatchNorm1d(hidden_dim),
                           nn.ReLU(inplace=True)]
            list_layers += [nn.Linear(hidden_dim, feat_dim),
                            nn.BatchNorm1d(feat_dim)]
        self.head = nn.Sequential(*list_layers)

    def forward(self, x):
        return self.head(x)
