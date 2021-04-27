#Parts of codes are copied and modified from "https://github.com/kuangliu/pytorch-cifar"

from __future__ import print_function

import numpy as np

import torch
import utils
import util
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as transforms
from util import progress_bar
from torch.utils.data import DataLoader
from torch.autograd import Variable

import pdb
import os
import argparse
import time

import struct
import random

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--se', default=0, type=int, help='start epoch')
parser.add_argument('--ne', default=0, type=int, help='number of epoch')
parser.add_argument('--pr', default=0, type=int, help='pruning') # mode=1 is pruning, mode=0 is no pruning
parser.add_argument('--bs', default=128, type=int, help='batch size')
parser.add_argument('--mode', default=0, type=int, help='train or inference') #mode=1 is train, mode=0 is inference
parser.add_argument('--network', default='ckpt_20200630_equalized.t0', help='input network ckpt name', metavar="FILE")
parser.add_argument('--output', default='garbage.txt', help='output file name', metavar="FILE")
parser.add_argument('--fixed', default=0, type=float)
parser.add_argument('--pprec', type=int, default=7, metavar='N',help='parameter precision for layer weight')
parser.add_argument('--aprec', type=int, default=7, metavar='N',help='arithmetic precision for convolution and fc')
parser.add_argument('--iwidth', type=int, default=6, metavar='N',help='integer bitwidth for internal part')


args = parser.parse_args()
torch.set_printoptions(precision=16)

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
test_loader = torch.utils.data.DataLoader(cifar_test,batch_size=192, shuffle=False,num_workers=6,drop_last=False)

mode = args.mode

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
		
		optimizer.step()

		train_loss += loss.data.item()
		_, predicted = torch.max(outputs.data, 1)
		total += float(targets.size(0))
		correct += predicted.eq(targets.data).cpu().sum().type(torch.FloatTensor)

		progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
			% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

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
	return acc

def retrain(net, epoch):
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
            net = util.quantize(net, args.pprec)

        optimizer.step()

        train_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += float(predicted.eq(targets.data).cpu().sum())

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        acc = 100.*correct/total
    if args.fixed:
        net = util.quantize(net, args.pprec)

print('==> Resuming from checkpoint..')
checkpoint = torch.load('./checkpoint/'+args.network)
net = checkpoint['net']
checkpoint = torch.load('./checkpoint/ckpt_20190913.t0')
net_origin = checkpoint['net']
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
net2 = MobileNetV2()
mask_null = util.maskGen(net, isbias=1, isempty = 1)
#mask_null = util.maskGen(net, isbias=0, isempty = 1)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

start_epoch = args.se
num_epoch = args.ne

if use_cuda:
	net.cuda()
	net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
	net2.cuda()
	net2 = torch.nn.DataParallel(net2, device_ids=range(torch.cuda.device_count()))
	net_origin.cuda()
	net_origin = torch.nn.DataParallel(net_origin, device_ids=range(torch.cuda.device_count()))
	cudnn.benchmark = True

params = util.paramsGet(net)
tmp = (params.data != 0).sum()
print("Ratio of nonzero value : ",tmp.item()/params.size()[0])
print("Number of nonzero value : ",tmp.item())
print("Number of value", params.size()[0])

net2 = util.netMaskMul(net2, mask_null, isbias = 1)
net2 = util.addNetwork(net2, net, isbias = 1)
net2 = util.swapBatch(net2, net)

if args.fixed:
    net = util.quantize(net, args.pprec)
    net2 = util.quantize(net2, args.pprec)
    net_origin = util.quantize(net_origin, args.pprec)

def calcDiff(tensor):
    pass

def evalMetric(net, net_origin):
    params_net = utils.paramsGet(net) 
    params_net_origin = utils.paramsGet(net_origin) 
    print(params_net.size())
    print(params_net_origin.size())

evalMetric()
