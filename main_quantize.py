#Parts of codes are copied and modified from "https://github.com/kuangliu/pytorch-cifar"

from __future__ import print_function

import numpy as np

import torch
import utils
import math
import pdb
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils import progress_bar

import os
import argparse

import struct
import random

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.2, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--fixed', default=0, type=int, help='fixed point arithmetic apply')
parser.add_argument('--se', default=0, type=int, help='start epoch')
parser.add_argument('--ne', default=80, type=int, help='number of epoch')
parser.add_argument('--nc', default=4, type=int, help='number of cycle')
parser.add_argument('--pr', default=50, type=int, help='pruning') # pruning ratio
parser.add_argument('--bs', default=256, type=int, help='batch size')
parser.add_argument('--mode', default=1, type=int, help='0-test net, 1-lottery hypothesis based pruning') #mode=1 is train, mode=0 is inference
parser.add_argument('--network', default='ckpt_20200824_MobileNetV2.t0', help='input network ckpt name', metavar="FILE")
parser.add_argument('--netsel', type=int, default=0, help='input network ckpt name', metavar="FILE")
parser.add_argument('--output', default='ckpt_20200419_MobileNetV2.t0', help='output file name', metavar="FILE")
parser.add_argument('--initparam', default='mask_init_param_20200419_MobileNetV2.t0', help='initial parameter .dat file name', metavar="FILE")
parser.add_argument('--pprec', type=int, default=16, metavar='N',help='parameter precision for layer weight')
parser.add_argument('--aprec', type=int, default=12, metavar='N',help='Arithmetic precision for internal arithmetic')
parser.add_argument('--iwidth', type=int, default=6, metavar='N',help='integer bitwidth for internal part')

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

train_loader = torch.utils.data.DataLoader(cifar_train,batch_size=args.bs, shuffle=True,num_workers=1,drop_last=False)
test_loader = torch.utils.data.DataLoader(cifar_test,batch_size=512, shuffle=False,num_workers=5,drop_last=False)

mode = args.mode

class QConv2d(nn.Conv2d):
    def forward(self, input):
        input = super().forward(input)
        if args.fixed == 1:
            return quant(input, args.iwidth, args.pprec)
        else:
            return input

class QBatchNorm2d(nn.BatchNorm2d):
    def forward(self, input):
        input = super().forward(input)
        if args.fixed == 1:
            return quant(input, args.iwidth, args.pprec)
        else:
            return input

class QLinear(nn.Linear):
    def forward(self, input):
        input = super().forward(input)
        if args.fixed == 1:
            return quant(input, args.iwidth, args.pprec)
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
        out = F.relu6(self.bn1(self.conv1(x)))
        out = F.relu6(self.bn2(self.conv2(out)))
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
        out = F.relu6(self.bn1(out))
        out = self.layers(out)
        out = F.relu6(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def quant(input, intbit, fracbit):
    input = torch.div(torch.round(torch.mul(input,2 ** fracbit)),2 ** fracbit)
    input = torch.clamp(input,-(2 ** intbit), 2 ** intbit - 2 ** (-fracbit))
    return input

# Load checkpoint.
if args.mode == 0:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('./checkpoint/'+args.network)
    net = checkpoint['net']
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    params = utils.paramsGet(net)
    tmp = (params.data != 0).sum()
    print("Ratio of nonzero value : ",tmp.item()/params.size()[0])
    print("Number of nonzero value : ",tmp.item())
    print("Number of value", params.size()[0])

criterion = nn.CrossEntropyLoss()

start_epoch = args.se
num_epoch = args.ne

# Training
def train(net, epoch, lr):
    print('\nEpoch: %d' % epoch)
    net.train()

    train_loss = 0
    correct = 0
    total = 0

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

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

def retrain(net, epoch, mask_prune, lr):
    print('\nEpoch: %d' % epoch)
    global best_acc

    net.train()
    train_loss = 0
    total = 0
    correct = 0

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        net = utils.pruneNetwork(net, mask_prune)

        if args.fixed:
            net = utils.quantize(net, args.pprec)

        optimizer.step()

        train_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += float(predicted.eq(targets.data).cpu().sum())

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        acc = 100.*correct/total

    net = utils.pruneNetwork(net, mask_prune)

    if args.fixed:
        net = utils.quantize(net, args.pprec)

def analyzeNet(net):
    layer_channel_range = []
    layer_depth_range = []
    #layer_num_filter = []
    #layer_filter_depth = []
    count = 0
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            layer_maximum_weight = torch.max(m.weight)
            layer_minimum_weight = torch.min(m.weight)
            #layer_num_filter.append(m.weight.size()[0])
            #layer_filter_depth.append(m.weight.size()[1])

            #finding channel range
            for i in range(m.weight.size()[0]):
                channel_maximum_weight = torch.max(m.weight[i])
                channel_minimum_weight = torch.min(m.weight[i])
                layer_channel_range.append(channel_maximum_weight - channel_minimum_weight)

            #finding range of specific depth of all filters
            for i in range(m.weight.size()[1]):
                depth_maximum_weight = torch.max(m.weight[:,i])
                depth_minimum_weight = torch.min(m.weight[:,i])
                layer_depth_range.append(depth_maximum_weight - depth_minimum_weight)
    return layer_channel_range, layer_depth_range

def equalizeLayer(net, layer_channel_range, layer_depth_range):
    layer_counter = 0
    list_counter = 0
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            if isresidual:
                pass
            else:
                for i in range(m.weight.size()[0]):
                    equalizing_param = layer_channel_range[list_counter]*layer_depth_range[i + m.weight.size()[1]]
                    m.weight.data[i] = m.weight.data[i]/equalizing_param 
                    list_counter += 1
            layer_counter += 1
                
    return net

## inference vs. Train+Inference
if mode == 0: # only inference
    test(net)

elif mode == 2: # mode=1 Lottery ticket hypothesis based training and pruning
    checkpoint = torch.load('./checkpoint/'+args.network)
    net = checkpoint['net']
    best_acc = checkpoint['acc']

    layer_channel_range, layer_depth_range = analyzeNet(net)
    print(len(layer_channel_range), len(layer_depth_range))
    exit()
    net = equalizeLayer(net, layer_channel_range, layer_depth_range)
    
else:
    pass

