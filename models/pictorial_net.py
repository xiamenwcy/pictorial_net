# coding: utf-8


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict


# 7x7 convolution
def conv7x7(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False)


# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


# 1x1 convolution
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)


# residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv1x1(in_channels, 32)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(32, 32, stride)
        self.bn2 = nn.BatchNorm2d(32, 32)
        self.conv3 = conv1x1(32, out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels, out_channels)
        self.downsample = downsample
        self.bn4 = nn.BatchNorm2d(out_channels, out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample:
            residual = self.downsample(x)
            residual = self.bn4(residual)
        out += residual
        out = self.relu(out)

        return out


# half-scale
class Hourglass_half_scale(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Hourglass_half_scale, self).__init__()

        self.conv1 =conv7x7(in_channels, 64, stride)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.residual_module1 = ResidualBlock(out_channels, out_channels)
        self.residual_module2 = ResidualBlock(out_channels, out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.residual_module1(out)
        out = self.residual_module2(out)

        return out


class Hourglass(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(Hourglass, self).__init__()

        self.residual_module1 = ResidualBlock(in_channels, in_channels, stride, conv1x1(in_channels, in_channels, stride))
        self.residual_module2 = ResidualBlock(in_channels, in_channels, stride, conv1x1(in_channels, out_channels, stride))
        self.residual_module3 = ResidualBlock(in_channels, in_channels, stride, conv1x1(in_channels, out_channels, stride))
        self.nearest1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.nearest2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.nearest3 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x_size = x.size()
        y1 = self.residual_module1(x)
        y1_size = y1.size()
        y2 = self.residual_module2(y1)
        y2_size = y2.size()
        y3 = self.residual_module3(y2)
        y4 = F.upsample(y3, y2_size[2:], mode='bilinear')
        y4 += y2
        y5 = F.upsample(y4, y1_size[2:], mode='bilinear')
        y5 +=y1
        y6 = F.upsample(y5, x_size[2:], mode='bilinear')
        out = y6 + x

        return out


class Hourglass_module(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(Hourglass_module, self).__init__()

        self.Hourglass = Hourglass(in_channels, out_channels, stride)
        self.conv1 = conv1x1(out_channels, out_channels)
        self.conv2 = conv1x1(out_channels, out_channels)
        self.conv_skip = conv1x1(out_channels, out_channels)

    def forward(self, x):
        y = self.Hourglass(x)
        gazemap = self.conv1(y)
        out = self.conv2(gazemap)
        y_out = self.conv_skip(y)
        out = out + y_out + x

        return out, gazemap


class Hourglass_net(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(Hourglass_net, self).__init__()

        self.Hourglass_half_scale = Hourglass_half_scale(in_channels, out_channels, 1)
        self.Hourglass_module1 = Hourglass_module(out_channels, out_channels, stride)
        self.Hourglass_module2 = Hourglass_module(out_channels, out_channels, stride)
        self.Hourglass_module3 = Hourglass_module(out_channels, out_channels, stride)

    def forward(self, x):
        y = self.Hourglass_half_scale(x)
        out, gazemap = self.Hourglass_module1(y)
        out, gazemap = self.Hourglass_module1(out)
        out, gazemap = self.Hourglass_module1(out)

        return out, gazemap


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class DenseNet(nn.Module):
    def __init__(self, growth_rate=8, block_config=(5, 5, 5, 5, 5), compression=0.5,
                 num_init_features=64, bn_size=4, drop_rate=0,
                 num_classes=2, small_inputs=True, efficient=False):

        super(DenseNet, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'
        self.avgpool_size = (2, 3)

        # First convolution
        if small_inputs:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(64, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ]))
            self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
            self.features.add_module('relu0', nn.ReLU(inplace=True))
            self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                                                           ceil_mode=False))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features * compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)

        # Final batch norm
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))


        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Initialization
        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                n = param.size(0) * param.size(2) * param.size(3)
                param.data.normal_().mul_(math.sqrt(2. / n))
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                param.data.fill_(0)
            elif 'classifier' in name and 'bias' in name:
                param.data.fill_(0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=self.avgpool_size).view(features.size(0), -1)
        out = self.classifier(out)
        return out


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        in_channels = 1
        out_channels = 64
        stride = 2

        self.Hourglass_net = Hourglass_net(in_channels, out_channels, stride)
        self.DenseNet = DenseNet(num_init_features=out_channels)

    def forward(self, x):
        y, gazemap = self.Hourglass_net(x)
        out = self.DenseNet(y)
        return out, gazemap


if __name__ == "__main__":
    net = Model()
    print(net)









