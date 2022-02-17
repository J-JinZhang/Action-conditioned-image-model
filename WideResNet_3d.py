 
"""
Author: Dr. Jin Zhang 
E-mail: j.zhang.vision@gmail.com
Created on 2022.02.01
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BasicBlock(nn.Module):
    expansion = 2

    def __init__(self, in_planes, out_planes, stride=1, convShortCut=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, out_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_planes)
        self.conv2 = nn.Conv3d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_planes)
        self.conv3 = nn.Conv3d(out_planes, out_planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.convShortCut = convShortCut
        self.stride = stride

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

        if self.convShortCut is not None:
            residual = self.convShortCut(x)

        out += residual
        out = self.relu(out)

        return out


class WideResNet(nn.Module):

    def __init__(self, block, layers, img_size=256, depth=10, k=1, num_classes=400, last_fc=False):
        self.last_fc = last_fc

        self.in_planes = 32
        super(WideResNet, self).__init__()
        self.conv1 = nn.Conv3d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 32 * k, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 64 * k, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128 * k, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256 * k, layers[3], stride=2)
        last_duration = math.ceil(depth / 16)
        last_size = math.ceil(img_size / 2**5)  # image size after a series of conv oprations
        self.avgpool = nn.AvgPool3d((1, last_size, last_size),)
        #self.fc = nn.Linear(512 * k * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, out_planes, num_layer, stride=1):
        convShortCut = None
        if stride != 1 or self.in_planes != out_planes * block.expansion:
            convShortCut = nn.Sequential(nn.Conv3d(self.in_planes, out_planes * block.expansion,
                                                  kernel_size=1, stride=stride, bias=False),
                                         nn.BatchNorm3d(out_planes * block.expansion))

        layers = []
        layers.append(block(self.in_planes, out_planes, stride, convShortCut))
        self.in_planes = out_planes * block.expansion
        for i in range(1, num_layer):
            layers.append(block(self.in_planes, out_planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        """if x.size(3) == 3:
            batch_size, length, depth, channels, height, width = x.size()
            x = x.permute(0, 1, 3, 2, 4, 5)
            x = x.reshape(batch_size*length, channels, depth, height, width)
        else:
            batch_size, depth, channels, height, width = x.size()
            x = x.permute(0, 2, 1, 3, 4)"""
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        """if x.size(3) == 3:
            x = x.view(batch_size, -1, 512)
        else:
            x = x.view(batch_size, -1)"""
        #if self.last_fc:
        #    x = self.fc(x)

        return x


def wide_resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = WideResNet(BasicBlock, [3, 4, 6, 3])
    return model
