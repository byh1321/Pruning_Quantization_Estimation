import argparse
import numpy as np


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--acc', default=0, type=float, help='original network accuracy')
parser.add_argument('--pr', default=0, type=float, help='pruning ratio(pr=90, 10% of param left)')
parser.add_argument('--quant', default=32, type=float, help='quantization bitwidth, for weight and activation')
parser.add_argument('--netsel', default=0, type=float, help='choose network to use')
parser.add_argument('--accdrop', default=1, type=float, help='choose accuracy drop threshold')
parser.add_argument('--flops', default=1, type=float, help='choose maximum flops threshold')
parser.add_argument('--channelscaling', default=1, type=float, help='channel scaling width')

args = parser.parse_args()

if netsel == 0:
    #VGG16 performance estimation variables
    acc_p = np.array([[0, 50, 60, 70, 80, 90, 95, 97, 98, 99],[1, 1, 0.997648686, 0.997510373, 0.989211618, 0.98077455, 0.949515906, 0.924481328, 0.883817427, 0.74329184]])
    acc_q = np.array([[32, 16, 15, 14, 13, 12, 11, 10, 9, 8],[1, 1, 1, 1, 1, 1, 0.996542185, 0.984094053, 0.973305671, 0.863070539]])
    acc_c = np.array([[1, 0.75, 0.5, 0.25],[1, 1, 0.956432, 0.912863]])
    f = open('MobileNetV2.txt','r')
    performance = f.readlines()


def interpolate(optimization_type, value):
    #interpolate 
    if value < 0:
        print("value is negative. check the parameter again.")
        exit()

    if optimization_type == 'pruning':
        array = value>acc_p[0]
        if np.sum(array) == 0:
            return 1
        else:
            idx = np.sum(array)-1
            range = acc_p[0,idx+1] - acc_p[0,idx]
            ratio = (acc_p[0,idx+1]-value)/range
            result = ratio * acc_p[1,idx] + (1-ratio) * acc_p[1,idx+1]
            return result

    elif optimization_type == 'quantization':
        array = value<acc_q[0]
        if np.sum(array) == 0:
            return 1
        elif np.sum(array) == acc_q.shape[1]:
            idx = np.sum(array)-1
            range = np.abs(acc_q[0,idx] - acc_q[0,idx-1])
            slope = (acc_q[1,idx] - acc_q[1,idx-1])/range
            extralength = acc_q[0,idx] - value
            result = acc_q[1,idx] + extralength*slope
        else:
            idx = np.sum(array)-1
            range = acc_q[0,idx+1] - acc_q[0,idx]
            ratio = (acc_q[0,idx+1]-value)/range
            result = ratio * acc_q[1,idx] + (1-ratio) * acc_q[1,idx+1]
            #print(result)
            return result
            
    elif optimization_type == 'channelscaling':
        array = value<acc_c[0]
        if np.sum(array) == 0:
            return 1
        elif np.sum(array) == acc_c.shape[1]:
            idx = np.sum(array)-1
            range = np.abs(acc_c[0,idx] - acc_c[0,idx-1])
            slope = (acc_c[1,idx] - acc_c[1,idx-1])/range
            extralength = acc_c[0,idx] - value
            result = acc_c[1,idx] + extralength*slope
        else:
            idx = np.sum(array)-1
            range = acc_c[0,idx+1] - acc_c[0,idx]
            ratio = (acc_c[0,idx+1]-value)/range
            result = ratio * acc_c[1,idx] + (1-ratio) * acc_c[1,idx+1]
            #print(result)
            return result

#def get_compentation_param(pr, quant, channelscaling):
#    pr = (100-pr)/100
#    quant = quant/32
#    information_left = pr * quant * channelscaling
#    if 


def main():
    pr = interpolate('pruning', args.pr)
    quant = interpolate('quantization', args.quant)
    channelscaling = interpolate('channelscaling', args.channelscaling)
    
    best_pr, best_quant, best_channelscaling, best_acc = get_best_combination
    #compensate_param = get_compentation_param(args.pr, args.quant, args.channelscaling)

    #print("Expected acc : ",args.acc * pr * quant * channelscaling)
    #print(args.quant, args.pr/100, args.acc * pow(pr * quant * channelscaling,1.2), end=", ")
    print(args.acc * pow(pr * quant * channelscaling,1.2), end=", ")

if __name__ == '__main__':
    main()
