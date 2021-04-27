import torch
import torch.nn as nn
import torch.nn.functional as F

iwidth = 6
pprec = 9

class QConv2d(nn.Conv2d):
    def forward(self, input):
        input = super().forward(input)
        if args.fixed == 1:
            return quant(input, iwidth, aprec)
        else:
            return input

class QBatchNorm2d(nn.BatchNorm2d):
    def forward(self, input):
        input = super().forward(input)
        if args.fixed == 1:
            return quant(input, iwidth, aprec)
        else:
            return input

class QLinear(nn.Linear):
    def forward(self, input):
        input = super().forward(input)
        if args.fixed == 1:
            return quant(input, iwidth, aprec)
        else:
            return input

class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = QConv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn1 = QBatchNorm2d(planes)
        self.conv2 = QConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=True)
        self.bn2 = QBatchNorm2d(planes)
        self.conv3 = QConv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn3 = QBatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                QConv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=True),
                QBatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=100):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = QConv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = QBatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = QConv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn2 = QBatchNorm2d(1280)
        self.linear = QLinear(1280, num_classes)
        #self.fileoutindex = 0

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def quant(input, intbit, fracbit):
    input = torch.div(torch.round(torch.mul(input,2 ** fracbit)),2 ** fracbit)
    input = torch.clamp(input,-(2 ** intbit), 2 ** intbit - 2 ** (-fracbit))
    return input
