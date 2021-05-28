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
from utils import progress_bar, AverageMeter, accuracy

import os
import argparse
import time

import struct
import random
import math

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--se', default=0, type=int, help='start epoch')
parser.add_argument('--ne', default=0, type=int, help='number of epoch')
parser.add_argument('--pr', default=0, type=int, help='pruning') # mode=1 is pruning, mode=0 is no pruning
parser.add_argument('--bs', default=128, type=int, help='batch size')
parser.add_argument('--mode', default=1, type=int, help='train or inference') #mode=1 is train, mode=0 is inference
parser.add_argument('--channelwidth', default=1, type=float, help='channel width multiplier')
parser.add_argument('--network', default='mobilenetv2-c5e733a8.pth', help='input network ckpt name', metavar="FILE")
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

global top1_acc
top1_acc = 0  # best test accuracy
top5_acc = 0  # best test accuracy

traindir = os.path.join('/home2/ImageNet/train')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_dataset = dset.ImageFolder(traindir,transforms.Compose([transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),normalize,]))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True,num_workers=8, pin_memory=False)

valdir = os.path.join('/home2/ImageNet/val')
val_loader = torch.utils.data.DataLoader(dset.ImageFolder(valdir,transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),normalize])),batch_size=128, shuffle=False,num_workers=8, pin_memory=False)

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

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        QConv2d(inp, oup, 3, stride, 1, bias=False),
        QBatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        QConv2d(inp, oup, 1, 1, 0, bias=False),
        QBatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                QConv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                QBatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                QConv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                QBatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                QConv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                QBatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                QConv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                QBatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                QConv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                QBatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=args.channelwidth):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = QLinear(output_channel, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, QConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, QBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, QLinear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def mobilenetv2(**kwargs):
    """
    Constructs a MobileNet V2 model
    """
    return MobileNetV2(**kwargs)

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
        top1_acc = checkpoint['top1_acc']
        top5_acc = checkpoint['top5_acc']
    else:
        net = mobilenetv2()
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
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-5)

start_epoch = args.se
num_epoch = args.ne

def train(epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    net.train()

    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda is not None:
            inputs, targets = inputs.cuda(), targets.cuda()

        # compute output
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        utils.adjust_learning_rate(optimizer, epoch, batch_idx, len(train_loader), args.ne, args.lr)

        if batch_idx % 200 == 0:
            batch_time.update(time.time() - end)
            end = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, batch_idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


def test():
    global top1_acc
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    net.eval()
    
    end = time.time()
    count = 0
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)

        outputs = net(inputs)

        loss = criterion(outputs, targets)

        prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
        
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time

        if batch_idx % 50 == 0:
            batch_time.update(time.time() - end)
            end = time.time()
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   batch_idx, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    # Save checkpoint.
    if top1.avg > top1_acc:
        if mode == 0:
            print('Acc : {}'.format(top1.avg))
            return
        else:
            print('Saving.. Acc : {}'.format(top1.avg))
            state = {
                'net': net.module if use_cuda else net,
                'top1_acc': top1.avg,
                'top5_acc': top5.avg,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/'+str(args.output))
            top1_acc = top1.avg

def retrain(net, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    net.train()

    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda is not None:
            inputs, targets = inputs.cuda(), targets.cuda()

        # compute output
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        if args.fixed:
            net = utils.quantize(net, args.pprec)

        pruneNetworkQ(net, mask_prune)

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % 200 == 0:
            batch_time.update(time.time() - end)
            end = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, batch_idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

    if args.fixed:
        net = utils.quantize(net, args.pprec)

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
        elif args.channelwidth == 0.875:
            f1 = open("MobileNetV2_0.875.txt","a+")
            f2 = open("MobileNetV2_0.875_compressed.txt","a+")
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

# inference vs. Train+Inference
if mode == 0: # only inference
    #printAcc(net)
    test()
    #summary(net, (3, 32, 32))

#elif mode == 1: # mode=1 is training & inference @ each epoch
#    for large_epoch in range(0,3):
#        print('learning rate : ',optimizer.param_groups[0]['lr'])
#        for epoch in range(start_epoch, start_epoch+num_epoch):
#            print("epoch : {}".format(epoch))
#            print(time.ctime())
#            train(epoch)
#            test()
#        #optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.1

elif mode == 1: # mode=1 is training & inference @ each epoch
    for epoch in range(start_epoch, start_epoch+num_epoch):
        print("epoch : {}".format(epoch))
        print(time.ctime())
        train(epoch)
        test()
        print('learning rate : ',optimizer.param_groups[0]['lr'])

elif mode == 2: # retrain for quantization and pruning
    for epoch in range(0,num_epoch):
        print("epoch : {}".format(epoch))
        print(time.ctime())
        retrain(net, epoch) 

        test()
    f = open('Accuracy.txt','a+')
    print('Channel width, pr, accuracy : '+str(args.channelwidth)+', '+str(args.pr)+', '+str(top1_acc), file=f)
    f.close()
else:
    pass
