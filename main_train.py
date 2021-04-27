#Parts of codes are copied and modified from "https://github.com/kuangliu/pytorch-cifar"

from __future__ import print_function

import numpy as np

import torch
import utils
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchsummary import summary
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils import progress_bar

import os
import argparse
import timeit

import struct
import random

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--se', default=0, type=int, help='start epoch')
parser.add_argument('--ne', default=0, type=int, help='number of epoch')
parser.add_argument('--pr', default=0, type=int, help='pruning') # mode=1 is pruning, mode=0 is no pruning
parser.add_argument('--bs', default=128, type=int, help='batch size')
parser.add_argument('--mode', default=1, type=int, help='train or inference') #mode=1 is train, mode=0 is inference
parser.add_argument('--channelwidth', default=1, type=float, help='learning rate')
parser.add_argument('--network', default='ckpt_20191223_VGG16.t0', help='input network ckpt name', metavar="FILE")
parser.add_argument('--netsel', type=int, default=0, help='input network ckpt name', metavar="FILE")
parser.add_argument('--fixed', type=int, default=0, metavar='N',help='fixed=0 - floating point arithmetic')
parser.add_argument('--pprec', type=int, default=15, metavar='N',help='parameter precision for layer weight')
parser.add_argument('--aprec', type=int, default=15, metavar='N',help='Arithmetic precision for internal arithmetic')
parser.add_argument('--iwidth', type=int, default=16, metavar='N',help='integer bitwidth for internal part')
parser.add_argument('--initparam', default='NULL', help='initial parameter .dat file name', metavar="FILE")
parser.add_argument('--output', default='garbage.t0', help='output file name', metavar="FILE")


torch.set_printoptions(precision=2)
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy

transform_train = transforms.Compose([transforms.RandomCrop(32,padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
transform_test = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

cifar_train = dset.CIFAR100("/home/yhbyun/Dataset/CIFAR100/", train=True, transform=transform_train, target_transform=None, download=True)
cifar_test = dset.CIFAR100("/home/yhbyun/Dataset/CIFAR100/", train=False, transform=transform_test, target_transform=None, download=True)

train_loader = torch.utils.data.DataLoader(cifar_train,batch_size=args.bs, shuffle=True,num_workers=8,drop_last=False)
test_loader = torch.utils.data.DataLoader(cifar_test,batch_size=625, shuffle=False,num_workers=8,drop_last=False)

mode = args.mode

def pruneNetworkQ(net, mask):
    index = 0
    for m in net.modules():
        if isinstance(m, QConv2d):
            if m.groups != 1:
                pass
            else:
                m.weight.grad.data = torch.mul(m.weight.grad.data,mask[index].cuda())
                m.weight.data = torch.mul(m.weight.data,mask[index].cuda())
            index += 1
        elif isinstance(m, QLinear):
            m.weight.grad.data = torch.mul(m.weight.grad.data,mask[index].cuda())
            m.weight.data = torch.mul(m.weight.data,mask[index].cuda())
            index += 1
    return net

class QConv2d(nn.Conv2d):
    def forward(self, input):
        input = super().forward(input)
        if args.fixed == 1:
            return quant(input, args.iwidth, args.aprec)
        else:
            return input

class QBatchNorm2d(nn.BatchNorm2d):
    def forward(self, input):
        input = super().forward(input)
        if args.fixed == 1:
            return quant(input, args.iwidth, args.aprec)
        else:
            return input

class QLinear(nn.Linear):
    def forward(self, input):
        input = super().forward(input)
        if args.fixed == 1:
            return quant(input, args.iwidth, args.aprec)
        else:
            return input

class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = QConv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = QBatchNorm2d(planes)
        self.conv2 = QConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = QBatchNorm2d(planes)
        self.conv3 = QConv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = QBatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                QConv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
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
        self.conv1 = QConv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = QBatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = QConv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
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

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.conv1 = nn.Sequential(
            QConv2d(3, 64, 3, padding=1, bias=False),  
            QBatchNorm2d(64),  
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            QConv2d(64, 64, 3, padding=1, bias=False),  
            QBatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool1 = nn.Sequential(
            nn.MaxPool2d(2, 2),  
        )
        self.conv3 = nn.Sequential(
            QConv2d(64, 128, 3, padding=1, bias=False),  
            QBatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            QConv2d(128, 128, 3, padding=1, bias=False),  
            QBatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.maxpool2 = nn.Sequential(
            nn.MaxPool2d(2, 2),  
        )
        self.conv5 = nn.Sequential(
            QConv2d(128, 256, 3, padding=1, bias=False),  
            QBatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.conv6 = nn.Sequential(
            QConv2d(256, 256, 3, padding=1, bias=False),  
            QBatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.conv7 = nn.Sequential(
            QConv2d(256, 256, 3, padding=1, bias=False),  
            QBatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.maxpool3 = nn.Sequential(
            nn.MaxPool2d(2, 2),  
        )
        self.conv8 = nn.Sequential(
            QConv2d(256, 512, 3, padding=1, bias=False),  
            QBatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.conv9 = nn.Sequential(
            QConv2d(512, 512, 3, padding=1, bias=False),  
            QBatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.conv10 = nn.Sequential(
            QConv2d(512, 512, 3, padding=1, bias=False),  
            QBatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.maxpool4 = nn.Sequential(
            nn.MaxPool2d(2, 2),  
        )
        self.conv11 = nn.Sequential(
            QConv2d(512, 512, 3, padding=1, bias=False),  
            QBatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.conv12 = nn.Sequential(
            QConv2d(512, 512, 3, padding=1, bias=False),  
            QBatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.conv13 = nn.Sequential(
            QConv2d(512, 512, 3, padding=1, bias=False),  
            QBatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.maxpool5 = nn.Sequential(
            nn.MaxPool2d(2, 2)  
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            QLinear(512, 512, bias=False),  
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(p=0.5),
            QLinear(512, 512, bias=False),  
            nn.ReLU(inplace=True),
        )
        self.fc3 = nn.Sequential(
            QLinear(512, 100, bias=False)  
        )
        self._initialize_weights()

    def compfeature(self, input):
        prout1 = input.view(1, -1)
        filter = torch.ge(prout1, 0.1)
        mfilter = filter.view(-1, 1)
        print(mfilter)
        return mfilter

    def forward(self, x):

        out1 = self.conv1(x)  

        out2 = self.conv2(out1)  
        out3 = self.maxpool1(out2)

        out4 = self.conv3(out3)  

        out5 = self.conv4(out4)  
        out6 = self.maxpool2(out5)

        out7 = self.conv5(out6)  

        out8 = self.conv6(out7)  

        out9 = self.conv7(out8)  
        out10 = self.maxpool3(out9)

        out11 = self.conv8(out10)  

        out12 = self.conv9(out11)  

        out13 = self.conv10(out12)  
        out14 = self.maxpool4(out13)

        out15 = self.conv11(out14)  

        out16 = self.conv12(out15)  

        out17 = self.conv13(out16)  
        out18 = self.maxpool5(out17)

        out19 = out18.view(out18.size(0), -1)

        out20 = self.fc1(out19)  

        out21 = self.fc2(out20)  

        out22 = self.fc3(out21)  

        return out22
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, QConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, QBatchNorm2d):
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, QLinear):
                nn.init.normal_(m.weight, 0, 0.01)

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18,self).__init__()
        self.conv1 = nn.Sequential(
            QConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            QBatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1),
        )
        self.layer1_basic1 = nn.Sequential(
            QConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            QBatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=False),
        )
        self.layer1_basic2 = nn.Sequential(
            QConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            QBatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )
        self.layer1_relu1 = nn.Sequential(
            nn.ReLU(inplace=False),
        )
        self.layer1_basic3 = nn.Sequential(
            QConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            QBatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=False),
        )
        self.layer1_basic4 = nn.Sequential(
            QConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            QBatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )
        self.layer1_relu2 = nn.Sequential(
            nn.ReLU(inplace=False),
        )

        self.layer2_basic1 = nn.Sequential(
            QConv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            QBatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=False),
        )
        self.layer2_downsample = nn.Sequential(
            QConv2d(64, 128, kernel_size=1, stride=2, bias=False),
            QBatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )
        self.layer2_basic2 = nn.Sequential(
            QConv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            QBatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )
        self.layer2_relu1 = nn.Sequential(
            nn.ReLU(inplace=False),
        )
        self.layer2_basic3 = nn.Sequential(
            QConv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            QBatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=False),
        )
        self.layer2_basic4 = nn.Sequential(
            QConv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            QBatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )
        self.layer2_relu2 = nn.Sequential(
            nn.ReLU(inplace=False),
        )
        self.layer3_basic1 = nn.Sequential(
            QConv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            QBatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=False),
        )
        self.layer3_downsample = nn.Sequential(
            QConv2d(128, 256, kernel_size=1, stride=2, bias=False),
            QBatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )
        self.layer3_basic2 = nn.Sequential(
            QConv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            QBatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )
        self.layer3_relu1 = nn.Sequential(
            nn.ReLU(inplace=False),
        )
        self.layer3_basic3 = nn.Sequential(
            QConv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            QBatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=False),
        )
        self.layer3_basic4 = nn.Sequential(
            QConv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            QBatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=False),
        )
        self.layer3_relu2 = nn.Sequential(
            nn.ReLU(inplace=False),
        )

        self.layer4_basic1 = nn.Sequential(
            QConv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            QBatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=False),
        )
        self.layer4_downsample = nn.Sequential(
            QConv2d(256, 512, kernel_size=1, stride=2, bias=False),
            QBatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )
        self.layer4_basic2 = nn.Sequential(
            QConv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            QBatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=False),
        )
        self.layer4_relu1 = nn.Sequential(
            nn.ReLU(inplace=False),
        )
        self.layer4_basic3 = nn.Sequential(
            QConv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            QBatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=False),
        )
        self.layer4_basic4 = nn.Sequential(
            QConv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            QBatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=False),
        )
        self.layer4_relu2 = nn.Sequential(
            nn.ReLU(inplace=False),
        )
        self.linear = nn.Sequential(
            QLinear(512, 100, bias=False)
        )
        self._initialize_weights()

    def forward(self,x):
        out = x.clone()
        out = self.conv1(out)

        residual = out

        out = self.layer1_basic1(out)

        out = self.layer1_basic2(out)

        out += residual
        out = self.layer1_relu1(out)
        residual = out

        out = self.layer1_basic3(out)

        out = self.layer1_basic4(out)

        out += residual
        out = self.layer1_relu2(out)
        residual = self.layer2_downsample(out)

        out = self.layer2_basic1(out)

        out = self.layer2_basic2(out)

        out += residual

        out = self.layer2_relu1(out)
        residual = out

        out = self.layer2_basic3(out)

        out = self.layer2_basic4(out)

        out += residual
        out = self.layer2_relu2(out)

        residual = self.layer3_downsample(out)

        out = self.layer3_basic1(out)

        out = self.layer3_basic2(out)

        out += residual
        out = self.layer3_relu1(out)

        residual = out

        out = self.layer3_basic3(out)

        out = self.layer3_basic4(out)

        out += residual
        out = self.layer3_relu2(out)

        residual = self.layer4_downsample(out)

        out = self.layer4_basic1(out)

        out = self.layer4_basic2(out)

        out = self.layer4_relu1(out)
        residual = out

        out = self.layer4_basic3(out)

        out = self.layer4_basic4(out)

        out += residual
        out = self.layer4_relu2(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, QConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, QBatchNorm2d):
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, QLinear):
                nn.init.normal_(m.weight, 0, 0.01)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BasicBlock, self).__init__()
        reduction = 0.5
        if 2 == stride:
            reduction = 1
        elif in_channels > out_channels:
            reduction = 0.25
            
        self.conv1 = QConv2d(in_channels, int(in_channels * reduction), 1, stride, bias=True)
        self.bn1   = QBatchNorm2d(int(in_channels * reduction))
        self.conv2 = QConv2d(int(in_channels * reduction), int(in_channels * reduction * 0.5), 1, 1, bias=True)
        self.bn2   = QBatchNorm2d(int(in_channels * reduction * 0.5))
        self.conv3 = QConv2d(int(in_channels * reduction * 0.5), int(in_channels * reduction), (1, 3), 1, (0, 1), bias=True)
        self.bn3   = QBatchNorm2d(int(in_channels * reduction))
        self.conv4 = QConv2d(int(in_channels * reduction), int(in_channels * reduction), (3, 1), 1, (1, 0), bias=True)
        self.bn4   = QBatchNorm2d(int(in_channels * reduction))
        self.conv5 = QConv2d(int(in_channels * reduction), out_channels, 1, 1, bias=True)
        self.bn5   = QBatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if 2 == stride or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                            QConv2d(in_channels, out_channels, 1, stride, bias=True),
                            QBatchNorm2d(out_channels)
            )
            
    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = F.relu(self.bn3(self.conv3(output)))
        output = F.relu(self.bn4(self.conv4(output)))
        output = F.relu(self.bn5(self.conv5(output)))
        output += F.relu(self.shortcut(input))
        output = F.relu(output)
        return output
    
class SqueezeNext(nn.Module):
    def __init__(self, width_x, blocks, num_classes):
        super(SqueezeNext, self).__init__()
        self.in_channels = 64
        self.conv1  = QConv2d(3, int(width_x * self.in_channels), 3, 1, 1, bias=True)     # For Cifar10
        #self.conv1  = QConv2d(3, int(width_x * self.in_channels), 3, 2, 1, bias=True)     # For Tiny-ImageNet
        self.bn1    = QBatchNorm2d(int(width_x * self.in_channels))
        self.stage1 = self._make_layer(blocks[0], width_x, 32, 1)
        self.stage2 = self._make_layer(blocks[1], width_x, 64, 2)
        self.stage3 = self._make_layer(blocks[2], width_x, 128, 2)
        self.stage4 = self._make_layer(blocks[3], width_x, 256, 2)
        self.conv2  = QConv2d(int(width_x * self.in_channels), int(width_x * 128), 1, 1, bias=True)
        self.bn2    = QBatchNorm2d(int(width_x * 128))
        self.linear = QLinear(int(width_x * 128), num_classes)
        
    def _make_layer(self, num_block, width_x, out_channels, stride):
        strides = [stride] + [1] * (num_block - 1)
        layers  = []
        for _stride in strides:
            layers.append(BasicBlock(int(width_x * self.in_channels), int(width_x * out_channels), _stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))
        output = self.stage1(output)
        output = self.stage2(output)
        output = self.stage3(output)
        output = self.stage4(output)
        output = F.relu(self.bn2(self.conv2(output)))
        output = F.avg_pool2d(output, 4)
        output = output.view(output.size(0), -1)
        output = self.linear(output)
        return output
    
def SqNxt_23_1x(num_classes):
    return SqueezeNext(1.0, [6, 6, 8, 1], num_classes)

def SqNxt_23_1x_v5(num_classes):
    return SqueezeNext(1.0, [2, 4, 14, 1], num_classes)

def SqNxt_23_2x(num_classes):
    return SqueezeNext(2.0, [6, 6, 8, 1], num_classes)

def SqNxt_23_2x_v5(num_classes):
    return SqueezeNext(2.0, [2, 4, 14, 1], num_classes)

def conv_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif class_name.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

def quant(input, intbit, fracbit):
    input = torch.div(torch.round(torch.mul(input,2 ** fracbit)),2 ** fracbit)
    input = torch.clamp(input,-(2 ** intbit), 2 ** intbit - 2 ** (-fracbit))
    return input

# Load checkpoint.
if args.mode == 0:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('./checkpoint/'+args.network)
    net = checkpoint['net']
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    params = utils.paramsGet(net)
    tmp = (params.data != 0).sum()
    print("Ratio of nonzero value : ",tmp.item()/params.size()[0])
    print("Number of nonzero value : ",tmp.item())
    print("Number of value", params.size()[0])

elif args.mode == 1:
    if args.resume:
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load('./checkpoint/'+args.network)
        net = checkpoint['net']
        best_acc = checkpoint['acc']
    else:
        if args.netsel == 0:
            print("Generating VGG16 network")
            net = VGG16()
        elif args.netsel == 1:
            print("Generating ResNet18 network")
            net = ResNet18()
        elif args.netsel == 2:
            print("Generating SqueezeNext network")
            net = SqNxt_23_2x_v5(100)
        elif args.netsel == 3:
            print("Generating MobileNetV2 network")
            net = MobileNetV2()
        else:
            print("netsel == ",args.netsel)
        best_acc = 0
    if args.channelwidth == 1:
        pass
    else:
        mask_channel = utils.setMask(net, args.channelwidth, 1)

#For pruning and quantization
elif args.mode == 2:
    checkpoint = torch.load('./checkpoint/'+args.network)
    net = checkpoint['net']
    params = utils.paramsGet(net)
    if args.channelwidth == 1:
        thres = utils.findThreshold(params, args.pr)
    else:
        thres = utils.findThreshold(params, 100-((100-args.pr)/100)*args.channelwidth*100)
    mask_prune = utils.getPruningMask(net, thres)

    if args.resume:
        print('==> Resuming from checkpoint..')
        best_acc = 0 
    else:
        best_acc = 0

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

start_epoch = args.se
num_epoch = args.ne

# Training
def train(net, epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        if args.fixed:
            net = utils.quantize(net, args.pprec)

        if args.channelwidth != 1:
            net = pruneNetworkQ(net, mask_channel)

        optimizer.step()

        train_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += float(targets.size(0))
        correct += predicted.eq(targets.data).cpu().sum().type(torch.FloatTensor)

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    if args.fixed:
        net = utils.quantize(net, args.pprec)

def test(net):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += float(targets.size(0))
        correct += predicted.eq(targets.data).cpu().sum().type(torch.FloatTensor)

        progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if args.mode>0:
        if acc > best_acc:

            state = {
                'net': net.module if use_cuda else net,
                'acc': acc,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            else:
                print('Saving..')
                torch.save(state, './checkpoint/'+str(args.output))
            best_acc = acc
    return acc

def retrain(net, epoch, mask):
    print('\nEpoch: %d' % epoch)
    global best_acc
    net.train()
    train_loss = 0
    total = 0
    correct = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        if args.fixed:
            net = utils.quantize(net, args.pprec)

        if args.pr != 0:
            net = pruneNetworkQ(net, mask)

        optimizer.step()

        train_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += float(predicted.eq(targets.data).cpu().sum())

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        acc = 100.*correct/total

def test_with_mask(net, mask):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += float(targets.size(0))
        correct += predicted.eq(targets.data).cpu().sum().type(torch.FloatTensor)

        progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:

        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        if args.mode == 0:
            pass
        else:
            print('Saving..')
            torch.save(state, './checkpoint/'+str(args.output))
        best_acc = acc

    return acc

def printAcc(net):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += float(targets.size(0))
        correct += predicted.eq(targets.data).cpu().sum().type(torch.FloatTensor)

    #print((args.pprec+1), (args.iwidth + args.aprec + 1), args.pr/100, (100.*correct/total).item(), end=", ")
    if args.netsel == 0:
        if args.channelwidth == 1:
            f1 = open("VGG16.txt","a+")
            f2 = open("VGG16_compressed.txt","a+")
        elif args.channelwidth == 0.75:
            f1 = open("VGG16_0.75.txt","a+")
            f2 = open("VGG16_0.75_compressed.txt","a+")
        elif args.channelwidth == 0.5:
            f1 = open("VGG16_0.5.txt","a+")
            f2 = open("VGG16_0.5_compressed.txt","a+")
        elif args.channelwidth == 0.25:
            f1 = open("VGG16_0.25.txt","a+")
            f2 = open("VGG16_0.25_compressed.txt","a+")
    elif args.netsel == 1:
        if args.channelwidth == 1:
            f1 = open("ResNet18.txt","a+")
            f2 = open("ResNet18_compressed.txt","a+")
        elif args.channelwidth == 0.75:
            f1 = open("ResNet18_0.75.txt","a+")
            f2 = open("ResNet18_0.75_compressed.txt","a+")
        elif args.channelwidth == 0.5:
            f1 = open("ResNet18_0.5.txt","a+")
            f2 = open("ResNet18_0.5_compressed.txt","a+")
        elif args.channelwidth == 0.25:
            f1 = open("ResNet18_0.25.txt","a+")
            f2 = open("ResNet18_0.25_compressed.txt","a+")
    elif args.netsel == 2:
        if args.channelwidth == 1:
            f1 = open("SqueezeNext.txt","a+")
            f2 = open("SqueezeNext_compressed.txt","a+")
        elif args.channelwidth == 0.75:
            f1 = open("SqueezeNext_0.75.txt","a+")
            f2 = open("SqueezeNext_0.75_compressed.txt","a+")
        elif args.channelwidth == 0.5:
            f1 = open("SqueezeNext_0.5.txt","a+")
            f2 = open("SqueezeNext_0.5_compressed.txt","a+")
        elif args.channelwidth == 0.25:
            f1 = open("SqueezeNext_0.25.txt","a+")
            f2 = open("SqueezeNext_0.25_compressed.txt","a+")
    elif args.netsel == 3:
        if args.channelwidth == 1:
            f1 = open("MobileNetV2.txt","a+")
            f2 = open("MobileNetV2_compressed.txt","a+")
        elif args.channelwidth == 0.75:
            f1 = open("MobileNetV2_0.75.txt","a+")
            f2 = open("MobileNetV2_0.75_compressed.txt","a+")
        elif args.channelwidth == 0.5:
            f1 = open("MobileNetV2_0.5.txt","a+")
            f2 = open("MobileNetV2_0.5_compressed.txt","a+")
        elif args.channelwidth == 0.25:
            f1 = open("MobileNetV2_0.25.txt","a+")
            f2 = open("MobileNetV2_0.25_compressed.txt","a+")
    print((args.channelwidth), (args.iwidth + args.aprec + 1), args.pr/100, (100.*correct/total).item(), file=f1)
    print((100.*correct/total).item(), end=", ", file=f2)
    f1.close()
    f2.close()

## inference vs. Train+Inference
if mode == 0: # only inference
    #printAcc(net)
    #test(net)
    summary(net, (3, 32, 32))

elif mode == 1: # mode=1 is training & inference @ each epoch
    for epoch in range(start_epoch, start_epoch+num_epoch):
        start_time = timeit.default_timer()
        train(net, epoch)
        test(net)
        terminate_time = timeit.default_timer()
        print(format(terminate_time - start_time)+"sec")

elif args.mode == 2:
    for epoch in range(0, args.ne):
        retrain(net, epoch, mask_prune)
        test_with_mask(net, mask_prune)
else:
    pass

