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
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils import progress_bar

import os
import argparse

import struct
import random
#import VGG16
#import ResNet18
#import MobileNetV2
#import SqueezeNext

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--se', default=0, type=int, help='start epoch')
parser.add_argument('--ne', default=0, type=int, help='number of epoch')
parser.add_argument('--pr', default=0, type=int, help='pruning') # mode=1 is pruning, mode=0 is no pruning
parser.add_argument('--bs', default=128, type=int, help='batch size')
parser.add_argument('--mode', default=1, type=int, help='train or inference') #mode=1 is train, mode=0 is inference
parser.add_argument('--network', default='ckpt_20191223_VGG16.t0', help='input network ckpt name', metavar="FILE")
parser.add_argument('--netsel', type=int, default=0, help='input network ckpt name', metavar="FILE")
parser.add_argument('--fixed', type=int, default=0, metavar='N',help='fixed=0 - floating point arithmetic')
parser.add_argument('--pprec', type=int, default=6, metavar='N',help='parameter precision for layer weight')
parser.add_argument('--initparam', default='NULL', help='initial parameter .dat file name', metavar="FILE")
parser.add_argument('--output', default='garbage.txt', help='output file name', metavar="FILE")


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
            net = VGG16.CNN()
        elif args.netsel == 1:
            print("Generating ResNet18 network")
            net = ResNet18.ResNet18()
        elif args.netsel == 2:
            print("Generating SqueezeNext network")
            net = SqueezeNext.SqNxt_23_2x_v5(100)
        elif args.netsel == 3:
            print("Generating MobileNetV2 network")
            net = MobileNetV2.MobileNetV2()
        elif args.netsel == 4:
            print("Generating SqueezeNext network")
            net = SqueezeNext2.SqNxt_23_1x_v5()
        else:
            print("netsel == ",args.netsel)
        best_acc = 0

#For pruning and quantization
elif args.mode == 2:
    checkpoint = torch.load('./checkpoint/'+args.network)
    net = checkpoint['net']
    params = utils.paramsGet(net)
    thres = utils.findThreshold(params, args.pr)
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
            net = utils.pruneNetwork(net, mask)

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


## inference vs. Train+Inference
if mode == 0: # only inference
    test(net)

elif mode == 1: # mode=1 is training & inference @ each epoch
    for epoch in range(start_epoch, start_epoch+num_epoch):
        train(net, epoch)
        test(net)

elif args.mode == 2:
    for epoch in range(0, args.ne):
        retrain(net, epoch, mask_prune)
        test_with_mask(net, mask_prune)
else:
    pass

