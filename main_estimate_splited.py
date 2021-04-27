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
#torch.set_printoptions(precision=20)

use_cuda = torch.cuda.is_available()
best_loss = 10000  # best test accuracy

transform_train = transforms.Compose([transforms.RandomCrop(32,padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
transform_test = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

acc_train = torch.Tensor(np.loadtxt('VGG16_train_splited.txt'))
train_dataset = acc_train[:,:-1]
train_target = acc_train[:,-1]
acc_test = torch.Tensor(np.loadtxt('VGG16_test_splited.txt'))
test_dataset = acc_test[:,:-1]
test_target = acc_test[:,-1]

mode = args.mode

#class TinyClassifier(nn.Module):
#    def __init__(self, num_classes=1):
#        super(TinyClassifier, self).__init__()
#        self.linear1 = nn.Linear(4, 32)
#        self.linear2 = nn.Linear(32, 64)
#        self.linear3 = nn.Linear(64, 256)
#        self.linear4 = nn.Linear(256, 1)
#
#        self._initialize_weights()
#
#    def forward(self, x):
#        out = F.relu(self.linear1(x))
#        out = F.relu(self.linear2(out))
#        out = F.relu(self.linear3(out))
#        out = self.linear4(out)
#        return out
#
#    def _initialize_weights(self):
#        for m in self.modules():
#            if isinstance(m, nn.Linear):
#                nn.init.normal_(m.weight, 0, 0.01)

class PruningAccEstimator(nn.Module):
    def __init__(self):
        super(PruningAccEstimator, self).__init__()
        self.linear1 = nn.Linear(2, 1)

        self._initialize_weights()

    def forward(self, x):
        out = F.relu(self.linear1(x))
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                m.weight.data = torch.abs(m.weight.data)

class QuantizationAccEstimator(nn.Module):
    def __init__(self):
        super(QuantizationAccEstimator, self).__init__()
        self.linear1 = nn.Linear(2, 1)

        self._initialize_weights()

    def forward(self, x):
        out = F.relu(self.linear1(x))
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                m.weight.data[0] = m.weight.data[0]-130000

if args.mode == 0:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('./checkpoint/'+args.network)
    net_p = checkpoint['net_p']
    net_q = checkpoint['net_q']
    net_p.cuda()
    net_q.cuda()
    print(net_p.state_dict())
    print(net_q.state_dict())
    net_p = torch.nn.DataParallel(net_p, device_ids=range(torch.cuda.device_count()))
    net_q = torch.nn.DataParallel(net_q, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'

criterion = nn.MSELoss()

num_epoch = args.ne

#def train(net_p, net_q, epoch):
#    print('\nEpoch: %d' % epoch)
#    net_p.train()
#    net_q.train()
#
#    train_loss = 0
#    correct = 0
#    total = 0
#
#    #optimizer_p = optim.SGD(net_p.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
#    #optimizer_q = optim.SGD(net_q.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
#    optimizer_p = optim.Adam(net_p.parameters(), lr=lr)
#    optimizer_q = optim.Adam(net_q.parameters(), lr=lr)
#    scheduler_p = optim.lr_scheduler.StepLR(optimizer_p, step_size=1, gamma= 0.99)       
#    scheduler_q = optim.lr_scheduler.StepLR(optimizer_q, step_size=1, gamma= 0.99)       
#
#    inputs, targets = train_dataset.cuda(), train_target.cuda()
#    inputs_p = torch.FloatTensor([[torch.exp(-xval/0.01507), 1] for xval in inputs[:,2]])
#    inputs_q = torch.FloatTensor([[torch.exp(-xval/0.0181), 1] for xval in inputs[:,1]])
#    #inputs = inputs[:,0] * (param_quant_a - param_quant_b * torch.pow(param_quant_c,inputs[:,1]))\
#    #          * (param_pr_a - param_pr_b * torch.pow(param_pr_c,inputs[:,2]))
#    optimizer_p.zero_grad()
#    optimizer_q.zero_grad()
#    scheduler_p.step(loss)
#    scheduler_q.step(loss)
#
#    #outputs = net(inputs).squeeze()
#    outputs_p = net_p(inputs_p).squeeze()
#    outputs_q = net_q(inputs_q).squeeze()
#
#    outputs = outputs_p * outputs_q
#    #print('Outputs_p : ',outputs_p)
#    print('Inputs_q : ',inputs_q)
#    print('Outputs_q : ',outputs_q)
#    print('Inputs_p : ',inputs_p)
#    print('Outputs_p : ',outputs_p)
#    print('Outputs : ',outputs)
#    print('Targets : ',targets)
#
#    loss = criterion(outputs, targets)
#    loss.backward()
#
#    optimizer_p.step()
#    optimizer_q.step()
#
#    train_loss = loss.data.item()
#    total = float(targets.size(0))
#
#    print('Train loss: %.3f'%(train_loss))


# Training
def train_p(net_p, epoch):
    print('\nEpoch: %d' % epoch)
    net_p.train()

    train_loss = 0
    correct = 0
    total = 0

    inputs, targets = train_dataset.cuda(), train_target[0:10].cuda()
    inputs_p = torch.FloatTensor([[torch.exp(-xval/0.01507), 1] for xval in inputs[0:10,2]])
    optimizer_p.zero_grad()

    #outputs = net(inputs).squeeze()
    outputs_p = net_p(inputs_p).squeeze()

    print('Outputs_p : ',outputs_p)
    print('Targets : ',targets)
    for param_group in optimizer_p.param_groups:
        print(param_group['lr'])

    loss = criterion(outputs_p, targets)
    loss.backward()

    optimizer_p.step()

    train_loss = loss.data.item()
    total = float(targets.size(0))

    print('Train loss: %.3f'%(train_loss))

def train_q(net_q, epoch):
    print('\nEpoch: %d' % epoch)
    net_q.train()

    train_loss = 0
    correct = 0
    total = 0

    inputs, targets = train_dataset.cuda(), train_target[10:21].cuda()
    inputs_q = torch.FloatTensor([[torch.exp(-xval/0.0181), 1] for xval in inputs[10:21,1]])
    #inputs = inputs[:,0] * (param_quant_a - param_quant_b * torch.pow(param_quant_c,inputs[:,1]))\
    #          * (param_pr_a - param_pr_b * torch.pow(param_pr_c,inputs[:,2]))
    optimizer_q.zero_grad()

    #outputs = net(inputs).squeeze()
    outputs_q = net_q(inputs_q).squeeze()

    print('Outputs_q : ',outputs_q)
    print('Targets : ',targets)
    for param_group in optimizer_q.param_groups:
        print(param_group['lr'])

    loss = criterion(outputs_q, targets)
    loss.backward()

    optimizer_q.step()

    train_loss = loss.data.item()
    total = float(targets.size(0))

    print('Train loss: %.3f'%(train_loss))

def test(net_p, net_q):
    global best_loss
    net_p.eval()
    net_q.eval()

    test_loss = 0
    correct = 0
    total = 0

    inputs, targets = test_dataset.cuda(), test_target.cuda()
    inputs_p = torch.FloatTensor([[torch.exp(-xval/0.01507), 1] for xval in inputs[:,2]])
    inputs_q = torch.FloatTensor([[torch.exp(-xval/0.0181), 1] for xval in inputs[:,1]])
    #inputs = inputs[:,0] * (param_quant_a - param_quant_b * torch.pow(param_quant_c,inputs[:,1]))\
    #          * (param_pr_a - param_pr_b * torch.pow(param_pr_c,inputs[:,2]))

    outputs_p = net_p(inputs_p).squeeze()
    outputs_q = net_q(inputs_q).squeeze()
    outputs = outputs_p * outputs_q
    loss = criterion(outputs, targets)

    test_loss = loss.data.item()
    total = float(targets.size(0))

    print('Test loss: %.3f'%(test_loss))
    if args.mode == 0:
        print('Outputs_p : ',outputs_p)
        print('Outputs_q : ',outputs_q)
        print('Outputs : ',outputs)
        print('Targets : ',targets)

    #Save checkpoint.
    if test_loss < best_loss:

        state = {
            'net_p': net_p.module if use_cuda else net_p,
            'net_q': net_q.module if use_cuda else net_p,
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
    test(net_p, net_q)

elif mode == 1: # mode=1 Lottery ticket hypothesis based training and pruning
    if args.resume:
        checkpoint = torch.load('./checkpoint/'+args.network)
        net_p = checkpoint['net_p']
        net_q = checkpoint['net_q']
        best_loss = checkpoint['loss']

    else:
        net_p = PruningAccEstimator()
        net_q = QuantizationAccEstimator()

    lr = args.lr

    net_p.cuda()
    net_q.cuda()
    net_p = torch.nn.DataParallel(net_p, device_ids=range(torch.cuda.device_count()))
    net_q = torch.nn.DataParallel(net_q, device_ids=range(torch.cuda.device_count()))

    optimizer_p = optim.SGD(net_p.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    optimizer_q = optim.SGD(net_q.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    #optimizer_p = optim.Adam(net_p.parameters())
    #optimizer_q = optim.Adam(net_q.parameters())
    scheduler_p = optim.lr_scheduler.StepLR(optimizer_p, step_size=400, gamma= 0.1)       
    scheduler_q = optim.lr_scheduler.StepLR(optimizer_q, step_size=400, gamma= 0.1)       


    for epoch in range(num_epoch): 
        #train(net_p, net_q, epoch)
        train_p(net_p, epoch)
        train_q(net_q, epoch)
        test(net_p, net_q)
        scheduler_p.step()
        scheduler_q.step()

else:
    pass

