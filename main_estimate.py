#Parts of codes are copied and modified from "https://github.com/kuangliu/pytorch-cifar"

from __future__ import print_function

import numpy as np

import torch
import utils
import math
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
parser.add_argument('--ne', default=80, type=int, help='number of epoch')
parser.add_argument('--bs', default=256, type=int, help='batch size')
parser.add_argument('--mode', default=1, type=int, help='0-test net, 1-lottery hypothesis based pruning') #mode=1 is train, mode=0 is inference
parser.add_argument('--network', default='ckpt_20200924_estimator.t0', help='input network ckpt name', metavar="FILE")
parser.add_argument('--output', default='ckpt_20200924_estimator.t0', help='output file name', metavar="FILE")

args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_loss = 10000  # best test accuracy

transform_train = transforms.Compose([transforms.RandomCrop(32,padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
transform_test = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

acc_train = torch.Tensor(np.loadtxt('VGG16_train.txt'))
train_dataset = acc_train[:,:-1]
train_target = acc_train[:,-1]
acc_test = torch.Tensor(np.loadtxt('VGG16_test.txt'))
test_dataset = acc_test[:,:-1]
test_target = acc_test[:,-1]

mode = args.mode

class TinyClassifier(nn.Module):
    def __init__(self, num_classes=1):
        super(TinyClassifier, self).__init__()
        self.linear1 = nn.Linear(3, 32)
        self.linear2 = nn.Linear(32, 64)
        self.linear3 = nn.Linear(64, 256)
        self.linear4 = nn.Linear(256, 1)

        self._initialize_weights()

    def forward(self, x):
        out = F.relu(self.linear1(x))
        out = F.relu(self.linear2(out))
        out = F.relu(self.linear3(out))
        out = self.linear4(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)

#class TinyClassifier(nn.Module):
#    def __init__(self, num_classes=1):
#        super(TinyClassifier, self).__init__()
#        self.linear1 = nn.Linear(3, 32)
#        self.linear2 = nn.Linear(32, 1)
#
#        self._initialize_weights()
#
#    def forward(self, x):
#        out = F.relu(self.linear1(x))
#        out = F.relu(self.linear2(out))
#        return out
#
#    def _initialize_weights(self):
#        for m in self.modules():
#            if isinstance(m, nn.Linear):
#                nn.init.normal_(m.weight, 0, 0.01)

if args.mode == 0:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('./checkpoint/'+args.network)
    net = checkpoint['net']
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'

criterion = nn.MSELoss()
#criterion = nn.CrossEntropyLoss()

num_epoch = args.ne

def train(net, epoch):
    print('\nEpoch: %d' % epoch)
    net.train()

    train_loss = 0
    correct = 0
    total = 0

    #optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adam(net.parameters())
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma= 0.99)       

    inputs, targets = train_dataset.cuda(), train_target.cuda()
    scheduler.step()
    optimizer.zero_grad()

    outputs = net(inputs).squeeze()

    print('Inputs : ',inputs)
    print('Outputs : ',outputs)
    print('Targets : ',targets)

    loss = criterion(outputs, targets)
    loss.backward()

    optimizer.step()

    train_loss = loss.data.item()
    total = float(targets.size(0))

    print('Train loss: %.3f'%(train_loss))


## Training
#def train_p(net_p, epoch):
#    print('\nEpoch: %d' % epoch)
#    net_p.train()
#
#    train_loss = 0
#    correct = 0
#    total = 0
#
#    optimizer = optim.SGD(net_p.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
#
#    inputs, targets = train_dataset.cuda(), train_target[0:20].cuda()
#    inputs_p = torch.FloatTensor([[torch.exp(-xval/0.01507), 1] for xval in inputs[0:20,2]])
#    optimizer.zero_grad()
#
#    #outputs = net(inputs).squeeze()
#    outputs_p = net_p(inputs_p).squeeze()
#
#    print('Outputs_p : ',outputs_p)
#    print('Targets : ',targets)
#
#    loss = criterion(outputs_p, targets)
#    loss.backward()
#
#    optimizer.step()
#
#    train_loss = loss.data.item()
#    total = float(targets.size(0))
#
#    print('Train loss: %.3f'%(train_loss))
#
#def train_q(net_q, epoch):
#    print('\nEpoch: %d' % epoch)
#    net_q.train()
#
#    train_loss = 0
#    correct = 0
#    total = 0
#
#    optimizer = optim.SGD(net_q.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
#
#    inputs, targets = train_dataset.cuda(), train_target[21:38].cuda()
#    inputs_q = torch.FloatTensor([[torch.exp(-xval/0.0181), 1] for xval in inputs[21:38,1]])
#    #inputs = inputs[:,0] * (param_quant_a - param_quant_b * torch.pow(param_quant_c,inputs[:,1]))\
#    #          * (param_pr_a - param_pr_b * torch.pow(param_pr_c,inputs[:,2]))
#    optimizer.zero_grad()
#
#    #outputs = net(inputs).squeeze()
#    outputs_q = net_q(inputs_q).squeeze()
#
#    print('Outputs_q : ',outputs_q)
#    print('Targets : ',targets)
#
#    loss = criterion(outputs_q, targets)
#    loss.backward()
#
#    optimizer.step()
#
#    train_loss = loss.data.item()
#    total = float(targets.size(0))
#
#    print('Train loss: %.3f'%(train_loss))

def test(net):
    global best_loss
    net.eval()

    test_loss = 0
    correct = 0
    total = 0

    inputs, targets = test_dataset.cuda(), test_target.cuda()

    outputs = net(inputs).squeeze()
    loss = criterion(outputs, targets)

    test_loss = loss.data.item()
    total = float(targets.size(0))

    print('Test loss: %.3f'%(test_loss))
    if args.mode == 0:
        print('Inputs : ',inputs)
        print('Outputs : ',outputs)
        print('Targets : ',targets)

    #Save checkpoint.
    if test_loss < best_loss:

        state = {
            'net': net.module if use_cuda else net,
            'loss': test_loss,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        if args.mode == 0:
            pass
        else:
            print('Saving..')
            print('Best loss : ',best_loss)
            torch.save(state, './checkpoint/'+str(args.output))
        best_loss = test_loss

    #return acc

## inference vs. Train+Inference
if mode == 0: # only inference
    test(net)

elif mode == 1: # mode=1 Lottery ticket hypothesis based training and pruning
    if args.resume:
        checkpoint = torch.load('./checkpoint/'+args.network)
        net = checkpoint['net']
        best_loss = checkpoint['loss']

    else:
        net = TinyClassifier()

    lr = args.lr

    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    for epoch in range(num_epoch): 
        train(net, epoch)
        test(net)

else:
    pass

