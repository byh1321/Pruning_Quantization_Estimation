'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.init as init

import scipy.misc
from scipy import ndimage
import numpy as np

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 40.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

##########################################################################
# Codes under this line is written by YH.Byun

def print_4Dtensor_to_png(tensor, filename):
    npimg = np.array(tensor,dtype=float)
    npimg = npimg.squeeze(0)
    scipy.misc.toimage(npimg).save(filename+".png")

def genblurkernel(sigma):
    order = 0
    radius = int(4 * float(sigma) + 0.5)
    kernel = scipy.ndimage.filters._gaussian_kernel1d(sigma, order, radius)
    return kernel
    
def setMask(net, area, val):
    mask = maskGen(net)
    for i in range(len(mask)):
        num_filter = mask[i].size()[0]
        depth = mask[i].size()[1]
        if len(mask[i].size()) == 2:
            if i == (len(mask)-1):
                mask[i][:,0:int(depth*area)] = val
                #print(mask[i].size())
                #print('0, ',depth*area)
            else:
                mask[i][0:int(num_filter*area),0:int(depth*area)] = val
                #print(mask[i].size())
                #print(num_filter*area,',',depth*area)
        elif len(mask[i].size()) == 4:
            if i == 0:
                mask[i][0:int(num_filter*area),:,:,:] = val
                #print(mask[i].size())
                #print(num_filter*area,',0,0,0')
            else:
                mask[i][0:int(num_filter*area),0:int(depth*area),:,:] = val
                #print(mask[i].size())
                #print(num_filter*area,',',depth*area,',0,0')
    
    return mask

def saveInitialParameter(net, initparam):
    net_param = []
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            net_param.append(m.weight.data)
        elif isinstance(m, nn.Linear):
            net_param.append(m.weight.data)
    torch.save(net_param, initparam)
    print("saving initial parameters")
    
def quantize(net, pprec):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data = torch.round(m.weight.data / (2 ** -(pprec))) * (2 ** -(pprec))
            m.weight.data = torch.clamp(m.weight.data, -1, 1 - 2**(-pprec))
        elif isinstance(m, nn.Linear):
            m.weight.data = torch.round(m.weight.data / (2 ** -(pprec))) * (2 ** -(pprec))
            m.weight.data = torch.clamp(m.weight.data, -1, 1 - 2**(-pprec))
    return net

def printLayers(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            print(m)
        elif isinstance(m, nn.Linear):
            print(m)

def maskGen(net, isbias=0, isempty = 1):
    mask = []
    if isempty:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                mask.append(torch.zeros(m.weight.data.size()))
                if isbias == 1:
                    mask.append(torch.zeros(m.bias.data.size()))
                #print(torch.zeros(m.weight.data.size()).size())
            elif isinstance(m, nn.Linear):
                mask.append(torch.zeros(m.weight.data.size()))
                if isbias == 1:
                    mask.append(torch.zeros(m.bias.data.size()))
                #print(torch.zeros(m.weight.data.size()).size())
    else:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                mask.append(torch.ones(m.weight.data.size()))
                if isbias == 1:
                    mask.append(torch.ones(m.bias.data.size()))
                #print(torch.ones(m.weight.data.size()).size())
            elif isinstance(m, nn.Linear):
                mask.append(torch.ones(m.weight.data.size()))
                if isbias == 1:
                    mask.append(torch.zeros(m.bias.data.size()))
                #print(torch.ones(m.weight.data.size()).size())
    return mask

def pruneNetwork(net, mask):
    index = 0
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.grad.data = torch.mul(m.weight.grad.data,mask[index].cuda())
            m.weight.data = torch.mul(m.weight.data,mask[index].cuda())
            index += 1
        elif isinstance(m, nn.Linear):
            m.weight.grad.data = torch.mul(m.weight.grad.data,mask[index].cuda())
            m.weight.data = torch.mul(m.weight.data,mask[index].cuda())
            index += 1
    return net

def pruneNetworkQ(net, mask):
    index = 0
    for m in net.modules():
        if isinstance(m, QConv2d):
            m.weight.grad.data = torch.mul(m.weight.grad.data,mask[index].cuda())
            m.weight.data = torch.mul(m.weight.data,mask[index].cuda())
            index += 1
        elif isinstance(m, QLinear):
            m.weight.grad.data = torch.mul(m.weight.grad.data,mask[index].cuda())
            m.weight.data = torch.mul(m.weight.data,mask[index].cuda())
            index += 1
    return net

def paramsGet(net):
    index = 0
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            if index == 0:
                params = m.weight.view(-1,)
                index += 1
            else:
                params = torch.cat((params,m.weight.view(-1,)),0)
                index += 1
        elif isinstance(m, nn.Linear):
            params = torch.cat((params,m.weight.view(-1,)),0)
            index += 1
    return params

def findThreshold(params, pr):
    thres=0
    while 1:
        tmp = (torch.abs(params.data)<thres).type(torch.FloatTensor)
        result = torch.sum(tmp)/params.size()[0]
        if (pr/100)<result:
            #print("threshold : {}".format(thres))
            return thres
        else:
            thres += 0.0001

#def findThreshold(params, pr):
#    params_sorted, indice = torch.sort(params)
#    index = int(pr * params.size()[0] / 100)
#    print(params_sorted[13228760])
#    print(params.size())
#    print(index)
#    return params_sorted[index].item()

def getPruningMask(net, thres):
    index = 0
    mask = []
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            mask.append((torch.abs(m.weight.data)>thres).type(torch.FloatTensor))
            index += 1
        elif isinstance(m, nn.Linear):
            mask.append((torch.abs(m.weight.data)>thres).type(torch.FloatTensor))
            index += 1
    return mask

def netMaskMul(net, mask, isbias=0, isbatch=0):
    index = 0
    if isbatch:
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data = torch.mul(m.weight.data,mask[index].cuda())
                index += 1
                m.bias.data = torch.mul(m.bias.data,mask[index].cuda())
                index += 1
    else:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data = torch.mul(m.weight.data,mask[index].cuda())
                index += 1
                if isbias == 1:
                    m.bias.data = torch.mul(m.bias.data,mask[index].cuda())
                    index += 1
            elif isinstance(m, nn.Linear):
                m.weight.data = torch.mul(m.weight.data,mask[index].cuda())
                index += 1
                if isbias == 1:
                    m.bias.data = torch.mul(m.bias.data,mask[index].cuda())
                    index += 1
    return net

def addNetwork(net, net2, isbias=0):
    index = 0
    mask = saveNetwork(net2, isbias)
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data = torch.add(m.weight.data,mask[index].cuda())
            index += 1
            if isbias:
                m.bias.data = torch.add(m.bias.data,mask[index].cuda())
                index += 1
        elif isinstance(m, nn.Linear):
            m.weight.data = torch.add(m.weight.data,mask[index].cuda())
            index += 1
            if isbias:
                m.bias.data = torch.add(m.bias.data,mask[index].cuda())
                index += 1
    return net

def netMaskAdd(net, mask, isbias=0, isbatch=0):
    index = 0
    if isbatch:
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data = torch.add(m.weight.data,mask[index].cuda())
                index += 1
                m.bias.data = torch.add(m.bias.data,mask[index].cuda())
                index += 1
    else:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data = torch.add(m.weight.data,mask[index].cuda())
                index += 1
                if isbias == 1:
                    m.bias.data = torch.add(m.bias.data,mask[index].cuda())
                    index += 1
            elif isinstance(m, nn.Linear):
                m.weight.data = torch.add(m.weight.data,mask[index].cuda())
                index += 1
                if isbias == 1:
                    m.bias.data = torch.add(m.bias.data,mask[index].cuda())
                    index += 1
    return net

def saveNetwork(net, isbias=0):
    mask = []
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            mask.append(m.weight.data)
            if isbias:
                mask.append(m.bias.data)
        elif isinstance(m, nn.Linear):
            mask.append(m.weight.data)
            if isbias:
                mask.append(m.bias.data)
    return mask

def saveBatch(net, isempty=0):
    mask = []
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            if isempty:
                mask.append(torch.zeros(m.weight.size()))
                mask.append(torch.zeros(m.bias.size()))
            else:
                mask.append(m.weight.data)
                mask.append(m.bias.data)
    return mask

def printLayerName(net):
    index = 0
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            print(index, " : Conv2d layer, ", m.weight.size())
            index += 1
        elif isinstance(m, nn.Linear):
            print(index, " : FC layer, ", m.weight.size())
            index += 1
        elif isinstance(m, nn.BatchNorm2d):
            print(index, " : BatchNorm2d layer, ", m.weight.size())
            index += 1
    return net

def freezeNetwork(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            for param in m.parameters():
                param.requires_grad = False
        elif isinstance(m, nn.Linear):
            for param in m.parameters():
                param.requires_grad = False
        elif isinstance(m, nn.BatchNorm2d):
            for param in m.parameters():
                param.requires_grad = False
    return net

def absorb_bn(module, bn_module):
    w = module.weight.data
    if module.bias is None:
        zeros = torch.Tensor(module.out_channels).zero_().type(w.type())
        module.bias = nn.Parameter(zeros)
    b = module.bias.data
    invstd = bn_module.running_var.clone().add_(bn_module.eps).pow_(-0.5)
    w.mul_(invstd.view(w.size(0), 1, 1, 1).expand_as(w))
    b.add_(-bn_module.running_mean).mul_(invstd)

    if bn_module.affine:
        w.mul_(bn_module.weight.data.view(w.size(0), 1, 1, 1).expand_as(w))
        b.mul_(bn_module.weight.data).add_(bn_module.bias.data)

    bn_module.register_buffer('running_mean', torch.zeros(module.out_channels).cuda())
    bn_module.register_buffer('running_var', torch.ones(module.out_channels).cuda())
    bn_module.register_parameter('weight', None)
    bn_module.register_parameter('bias', None)
    bn_module.affine = False

def is_bn(m):
    return isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)


def is_absorbing(m):
    return isinstance(m, nn.Conv2d)  or isinstance(m, nn.Linear)

def search_absorbe_bn(model):
    prev = None
    for m in model.children():
        if is_bn(m) and is_absorbing(prev):
            m.absorbed = True
            absorb_bn(prev, m)
        search_absorbe_bn(m)
        prev = m

#swap bias in net with bias in net2
def swapBias(net, net2):
    mask_bias = saveBias(net2)
    mask_bias_null = saveBias(net2, isempty=1)
    index = 0
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.bias.data = torch.mul(m.bias.data,mask_bias_null[index].cuda())
            m.bias.data = torch.add(m.bias.data,mask_bias[index].cuda())
            index += 1
        elif isinstance(m, nn.Linear):
            m.bias.data = torch.mul(m.bias.data,mask_bias_null[index].cuda())
            m.bias.data = torch.add(m.bias.data,mask_bias[index].cuda())
            index += 1
    return net

def saveBias(net, isempty=0):
    mask = []
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            if isempty:
                mask.append(torch.zeros(m.bias.data.size()))
            else:
                mask.append(m.bias.data)
        elif isinstance(m, nn.Linear):
            if isempty:
                mask.append(torch.zeros(m.bias.data.size()))
            else:
                mask.append(m.bias.data)
    return mask

def concatMask(mask1, mask2):
    index = 0
    for i in range(len(mask1)):
        mask1[index] = ((mask1[index] + mask2[index]) != 0).type(torch.FloatTensor)
        index += 1
    return mask1

def getExtendedMask(mask):
    index = torch.FloatTensor()
    for i in range(len(mask)):
        if mask[i].dim() == 4:
            mask_size = mask[i].size()[0] * mask[i].size()[1] * mask[i].size()[2] * mask[i].size()[3]
            if mask[i].size()[2] == 1:
                if mask[i].size()[1] % 3 == 1:
                    index_for_print = torch.zeros(mask[i].size()[0], mask[i].size()[1]+2,1,1)
                    index_for_print[:,:-2,:,:] = mask[i].data
                elif mask[i].size()[1] % 3 == 2:
                    index_for_print = torch.zeros(mask[i].size()[0], mask[i].size()[1]+1,1,1)
                    index_for_print[:,:-1,:,:] = mask[i].data
                else:
                    index_for_print = mask[i].data
                index_for_print = index_for_print.view(-1,3)
                index_for_print = (torch.sum(index_for_print, dim=1) != 0).type(torch.FloatTensor)
                index = torch.cat((index, index_for_print),0)
            else:
                index_for_print = mask[i].data
                index_for_print = index_for_print.view(-1,3)
                index_for_print = (torch.sum(index_for_print, dim=1) != 0).type(torch.FloatTensor)
                index = torch.cat((index, index_for_print),0)
        else:
            mask_size = mask[i].size()[0] * mask[i].size()[1]
            index_for_print = torch.zeros(mask[i].size()[0], mask[i].size()[1] + 1)
            index_for_print[:,:-1] = mask[i].data
            index_for_print = index_for_print.view(-1,3)
            index_for_print = (torch.sum(index_for_print, dim=1) != 0).type(torch.FloatTensor)
            index = torch.cat((index, index_for_print),0)
    return index

def quantBatch(net, intbit, pprec):
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.running_var.data = torch.round(m.running_var.data / (2 ** -(pprec))) * (2 ** -(pprec))
            m.running_var.data = torch.clamp(m.running_var.data, max=1, min=2**(-intbit))
            m.weight.data = torch.round(m.weight.data / (2 ** -(15))) * (2 ** -(15))
            m.weight.data = torch.clamp(m.weight.data,-(2) ** intbit, 2 ** intbit)
            m.bias.data = torch.round(m.bias.data / (2 ** -(pprec))) * (2 ** -(pprec))
            m.bias.data = torch.clamp(m.bias.data,-(2) ** intbit, 2 ** intbit)
            m.running_mean.data = torch.round(m.running_mean.data / (2 ** -(pprec))) * (2 ** -(pprec))
            m.running_mean.data = torch.clamp(m.running_mean.data,-(2) ** intbit, 2 ** intbit)
    return net

def swapBiasandBatch(net, net2):
    mask_bias = saveBias(net2, isbatch=1)
    mask_bias_null = saveBias(net2, isempty=1, isbatch=1)
    index = 0
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.bias.data = torch.mul(m.bias.data,mask_bias_null[index].cuda())
            m.bias.data = torch.add(m.bias.data,mask_bias[index].cuda())
            index += 1
        elif isinstance(m, nn.Linear):
            m.bias.data = torch.mul(m.bias.data,mask_bias_null[index].cuda())
            m.bias.data = torch.add(m.bias.data,mask_bias[index].cuda())
            index += 1
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data = torch.mul(m.weight.data,mask_weight_null[index].cuda())
            m.weight.data = torch.add(m.weight.data,mask_weight[index].cuda())
            m.bias.data = torch.mul(m.bias.data,mask_bias_null[index].cuda())
            m.bias.data = torch.add(m.bias.data,mask_bias[index].cuda())
            m.running_mean.data = torch.mul(m.running_mean.data,mask_running_mean_null[index].cuda())
            m.running_mean.data = torch.add(m.running_mean.data,mask_running_mean[index].cuda())
            m.running_var.data = torch.mul(m.running_var.data,mask_running_var_null[index].cuda())
            m.running_var.data = torch.add(m.running_var.data,mask_running_var[index].cuda())
    return net

def swapBatch(net, net2):
    mask_batch = saveBatch(net2)
    mask_batch_null = saveBatch(net2, isempty=1)
    index = 0
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data = torch.mul(m.weight.data,mask_batch_null[index].cuda())
            m.weight.data = torch.add(m.weight.data,mask_batch[index].cuda())
            index += 1
            m.bias.data = torch.mul(m.bias.data,mask_batch_null[index].cuda())
            m.bias.data = torch.add(m.bias.data,mask_batch[index].cuda())
            index += 1
            m.running_mean.data = torch.mul(m.running_mean.data,mask_batch_null[index].cuda())
            m.running_mean.data = torch.add(m.running_mean.data,mask_batch[index].cuda())
            index += 1
            m.running_var.data = torch.mul(m.running_var.data,mask_batch_null[index].cuda())
            m.running_var.data = torch.add(m.running_var.data,mask_batch[index].cuda())
            index += 1
    return net

def saveBatch(net, isempty=0):
    mask = []
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            if isempty:
                mask.append(torch.zeros(m.weight.data.size()))
                mask.append(torch.zeros(m.bias.data.size()))
                mask.append(torch.zeros(m.running_mean.data.size()))
                mask.append(torch.zeros(m.running_var.data.size()))
            else:
                mask.append(m.weight.data)
                mask.append(m.bias.data)
                mask.append(m.running_mean.data)
                mask.append(m.running_var.data)
    return mask

def printFeature(feature, filename):
    f = open(filename, 'w')
    for i in range(feature.data.size()[1]):
        for j in range(feature.data.size()[2]):
            for k in range(feature.data.size()[3]):
                print(feature.data[0,i,j,k].item(), file=f, end=',')
            print('',file=f)
        print('',file=f)
    f.close()
    return

def printconv1_0(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            print(m.weight[0])
            try:
                print(m.bias[0])
            except:
                print("There is no bias")
                pass
            return

def printbatch1(net):
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            print(m.weight)
            print(m.bias)
            print(m.running_mean)
            print(m.running_var)
            return

def printlinear1_0(net):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            print(m.weight[0])
            try:
                print(m.bias[0])
            except:
                print("There is no bias")
                pass
            return


def float_to_hex(float_):
    temp = float_ * 2**7  # Scale the number up.
    temp = torch.round(temp)     # Turn it into an integer.
    temp = int(temp)
    temp = temp & 0xff
    return '{:02x}'.format(temp)

def float_to_hex_16(float_):
    temp = float_ * 2**8  # Scale the number up.
    temp = torch.round(temp)     # Turn it into an integer.
    temp = int(temp)
    temp = temp & 0xffff
    return '{:04x}'.format(temp)

