import argparse
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--netsel', default=0, type=float, help='choose network to use')
parser.add_argument('--accdrop', default=0, type=float, help='choose accuracy drop threshold')
#parser.add_argument('--pparam', default=0, type=float, help='expected percent of remaining parameter')
parser.add_argument('--pparam', default=0, type=float, help='choose percent of remaining parameter')
#parser.add_argument('--powerc', default=1, type=float, help='power of degrading term for channel scaling')
#parser.add_argument('--powerq', default=1, type=float, help='power of degrading term for quantization')
#parser.add_argument('--powerp', default=1, type=float, help='power of degrading term for pruning')
parser.add_argument('--actual', default=0, type=float, help='actual = 1, find actual optimal point')
parser.add_argument('--printprocess', default=0, type=float, help='printprocess = 1, print procedures (for debug)')
parser.add_argument('--thres', default=1, type=float, help='allowed maximum accuracy difference between estimated acc and evaluated acc')
parser.add_argument('--unified', default=0, type=int, help='use unified factor search')
parser.add_argument('--ngreedy', default=10, type=int, help='number of greedy algorithm search')

args = parser.parse_args()

global performance_estimated
performance_estimated = []

if args.netsel == 0:
    #VGG16 performance estimation variables
    acc_c = np.array([[1, 0.75, 0.5, 0.25],[1, 1, 0.956432, 0.86667]]) # shape 2,4
    acc_q = np.array([[32, 16, 15, 14, 13, 12, 11, 10, 9, 8],[1, 1, 1, 0.999723358, 0.999308437, 0.998201936, 0.996957148, 0.983817389, 0.973443945, 0.863070561]])
    acc_p = np.array([[0, 50, 60, 70, 80, 90, 95, 97, 98, 99],[1, 1, 0.99778699, 0.997510411, 0.989349914, 0.980912838, 0.949515927, 0.924619619, 0.883817448, 0.743291863]])
    net_acc = 72.3
    nparam_net = 15285952
elif args.netsel == 1:
    #ResNet18 performance estimation variables
    acc_c = np.array([[1, 0.75, 0.5, 0.25],[1, 0.988334479, 0.957413932, 0.899367538]]) # shape 2,4
    acc_q = np.array([[32, 16, 15, 14, 13, 12, 11, 10, 9, 8],[1, 1, 1, 1, 0.998453962, 0.997470129, 0.991426555, 0.990442723, 0.988615528, 0.945326784]])
    acc_p = np.array([[0, 50, 60, 70, 80, 90, 95, 97, 98, 99],[1, 0.999297216, 0.998313384, 0.995502464, 0.992832015, 0.986366814, 0.95895997, 0.933661263, 0.905411058, 0.77919884]])
    net_acc = 71.15
    nparam_net = 11210432
elif args.netsel == 2:
    #SqueezeNext performance estimation variables
    acc_c = np.array([[1, 0.75, 0.5, 0.25],[1, 0.986339619, 0.956230841, 0.842626186]]) # shape 2,4
    acc_q = np.array([[32, 16, 15, 14, 13, 12, 11, 10, 9, 8],[1, 0.999721262, 0.999721262, 0.998187939, 0.99498187, 0.967382264, 0.919152557, 0.914831324, 0.574714257, 0.093114023]])
    acc_p = np.array([[0, 50, 60, 70, 80, 90, 95, 97, 98, 99],[1, 0.994284973, 0.994703133, 0.99233349, 0.98773352, 0.959158065, 0.900892099, 0.830777812, 0.765681655, 0.521605789]])
    net_acc = 71.74
    nparam_net = 3076480
    
elif args.netsel == 3:
    #MobileNetV2 performance estimation variables
    acc_c = np.array([[1, 0.75, 0.5, 0.25],[1, 0.989623231, 0.938967759, 0.906608458]]) # shape 2,4
    acc_q = np.array([[32, 16, 15, 14, 13, 12, 11, 10, 9, 8],[1, 0.999044242, 0.999044242, 0.993309696, 0.990715452, 0.972829084, 0.820999492, 0.819634087, 0.181048616, 0.016657565]])
    acc_p = np.array([[0, 50, 60, 70, 80, 90, 95, 97, 98, 99],[1, 1, 1, 0.999044242, 0.99440202, 0.937738972, 0.79396509, 0.667258334, 0.523620967, 0.099126167]])
    net_acc = 73.24
    nparam_net = 2377024

if args.netsel == 0:
    performance = np.loadtxt('VGG16.txt')
if args.netsel == 1:
    performance = np.loadtxt('ResNet18.txt')
if args.netsel == 2:
    performance = np.loadtxt('SqueezeNext.txt')
if args.netsel == 3:
    performance = np.loadtxt('MobileNetV2.txt')

def print_estimation(filename=None):
    global performance_estimated
    performance_estimated = np.array(performance_estimated)
    if filename==None:
        for i in range(acc_c.shape[1]):
            for j in range(acc_q.shape[1]):
                for k in range(acc_p.shape[1]):
                    print("{:.3f}".format(performance_estimated[i*acc_q.shape[1]*acc_p.shape[1]+j*acc_p.shape[1]+k,3]), end='\t')
                print('')
            print('')
    else:
        f = open(filename,'w')
        for i in range(acc_c.shape[1]):
            for j in range(acc_q.shape[1]):
                for k in range(acc_p.shape[1]):
                    print(performance_estimated[i*acc_q.shape[1]*acc_p.shape[1]+j*acc_p.shape[1]+k,3], file=f, end='\t')
                print('',file=f)
            print('',file=f)
        f.close()
    return

def interpolate(optimization_type, value):
    #interpolate 
    if value < 0:
        print("value is negative. check the parameter again.")
        exit()

    if optimization_type == 'cs':
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

def get_best_combination_acc_based(power_c, power_q, power_p):
    global performance_estimated
    if args.netsel == 0:
        nparam_net = 15285952
    if args.netsel == 1:
        nparam_net = 11210432
    if args.netsel == 2:
        nparam_net = 3076480
    if args.netsel == 3:
        nparam_net = 2377024

    if args.pparam != 0:
        best_acc_est = 0
    else:
        best_nparam = 100000000
    for factor_c in acc_c[0]:
        for factor_q in acc_q[0]:
            for factor_p in acc_p[0]:
                idx_degrad_c = np.where(acc_c[0,:] == factor_c)[0][0]
                idx_degrad_q = np.where(acc_q[0,:] == factor_q)[0][0]
                idx_degrad_p = np.where(acc_p[0,:] == factor_p)[0][0]
                if args.actual:
                    acc = get_actual_acc(factor_c, factor_q, factor_p)
                else:
                    #acc = get_actual_acc(factor_c, factor_q, factor_p)
                    acc = net_acc * pow(acc_c[1,idx_degrad_c],power_c) * pow(acc_q[1,idx_degrad_q],power_q) * pow(acc_p[1,idx_degrad_p],power_p)
                performance_estimated.append([factor_c, factor_q, factor_p, acc])
                nparam = nparam_net * pow(factor_c,2) * factor_q/32 * (100-factor_p)/100
                if args.pparam != 0:
                    if (args.pparam/100*nparam_net >= nparam) and (acc >= best_acc_est):
                        best_nparam = nparam
                        best_acc_est = acc
                        best_c = factor_c
                        best_q = factor_q
                        best_p = factor_p
                else:
                    if (net_acc - args.accdrop <= acc) and (best_nparam >= nparam):
                        best_nparam = nparam
                        best_acc_est = acc
                        best_c = factor_c
                        best_q = factor_q
                        best_p = factor_p

    performance_estimated = np.array(performance_estimated)
    return best_c, best_q, best_p, best_nparam, best_acc_est

#def get_best_combination_size_based(power_c, power_q, power_p):
#    global performance_estimated
#    if args.netsel == 0:
#        nparam_net = 15285952
#    if args.netsel == 1:
#        nparam_net = 11210432
#    if args.netsel == 2:
#        nparam_net = 3076480
#    if args.netsel == 3:
#        nparam_net = 2377024
#
#    if args.nparam != 0:
#        best_size = 0
#    else:
#        best_nparam = 100000000
#    for i in range(acc_c.shape[1]):
#        for j in range(acc_q.shape[1]):
#            for k in range(acc_p.shape[1]):
#                acc = net_acc * pow(acc_c[1,i],power_c) * pow(acc_q[1,j],power_q) * pow(acc_p[1,k],power_p)
#                performance_estimated.append([acc_c[0,i],acc_q[0,j],acc_p[0,k],acc])
#                nparam = nparam_net * pow(acc_c[0,i],2) * acc_q[0,j]/32 * (100-acc_p[0,k])/100
#                if args.nparam != 0:
#                    if (args.nparam >= nparam) and (acc >= best_acc):
#                        best_nparam = nparam
#                        best_acc = acc
#                        best_c = acc_c[0,i]
#                        best_q = acc_q[0,j]
#                        best_p = acc_p[0,k]
#                else:
#                    if (net_acc - args.accdrop <= acc) and (best_nparam >= nparam):
#                        best_nparam = nparam
#                        best_acc = acc
#                        best_c = acc_c[0,i]
#                        best_q = acc_q[0,j]
#                        best_p = acc_p[0,k]
#
#    performance_estimated = np.array(performance_estimated)
#    return best_c, best_q, best_p, best_nparam, best_acc

def get_actual_acc(best_c, best_q, best_p):
    rows1 = np.where(performance[:,0] == best_c)
    rows2 = np.where(performance[:,1] == best_q)
    rows3 = np.where(performance[:,2] == best_p/100)
    row = np.intersect1d(np.intersect1d(rows1, rows2),rows3)
    actual_acc = performance[row[0],3]
    return actual_acc

#def compensateEstimation(best_c, best_q, best_p, best_acc):
#    actual_acc = []
#    rows1 = np.where(performance[:,0] == best_c)
#    rows2 = np.where(performance[:,1] == best_q)
#    rows3 = np.where(performance[:,2] == best_p/100)
#    row = np.intersect1d(np.intersect1d(rows1, rows2),rows3)
#    actual_acc.append([best_c,best_q,best_p,performance[row[0],3]])
#    actual_acc = np.array(actual_acc)
#
#    performance_search = []
#    for i in np.arange(0.9, 1.5, 0.01):
#        for j in np.arange(0.9, 1.5, 0.01):
#            for k in np.arange(0.9, 1.5, 0.01):
#                acc = net_acc * pow(acc_c[1,l],power_c) * pow(acc_q[1,m],power_q) * pow(acc_p[1,n],power_p)
#                performance_search.append([acc_c[0,l],acc_q[0,m],acc_p[0,n],acc])
#    return power_c, power_q, power_p

def modify_power_factors(best_c, best_q, best_p):
    #modified_power_c = net_acc * pow(acc_c[1,i],power_c) * pow(acc_q[1,j],power_q) * pow(acc_p[1,k],power_p)
    idx_c = np.where(acc_c[0,:] == best_c)[0][0]
    idx_p = np.where(acc_p[0,:] == best_p)[0][0]
    idx_q = np.where(acc_q[0,:] == best_q)[0][0]
    #modified_power_c = get_actual_acc(best_c, best_q, 0) / net_acc \
    #        / np.log(acc_c[1,idx_c] * np.log(acc_c[1,idx_c] * net_acc)
    if acc_p[1,idx_p] == 1:
        modified_power_p = 1
    else:
        modified_power_p = np.log(get_actual_acc(best_c, 32, best_p)/get_actual_acc(best_c, 32, 0))/np.log(acc_p[1,idx_p])

    if acc_q[1,idx_q] == 1:
        modified_power_q = 1
    else:
        modified_power_q = np.log(get_actual_acc(best_c, best_q, best_p)/get_actual_acc(best_c, 32, best_p))/np.log(acc_q[1,idx_q])

    if args.printprocess:
        print("modified power_q : ", modified_power_q)
        print("modified power_p : ", modified_power_p)
    return 1, modified_power_q, modified_power_p

def modify_power_factors_unified(best_c, best_q, best_p):
    idx_c = np.where(acc_c[0,:] == best_c)[0][0]
    idx_p = np.where(acc_p[0,:] == best_p)[0][0]
    idx_q = np.where(acc_q[0,:] == best_q)[0][0]
    modified_power = np.log(get_actual_acc(best_c, best_q, best_p)/net_acc)/np.log(acc_c[1,idx_c]*acc_q[1,idx_q]*acc_p[1,idx_p])

    return modified_power, modified_power, modified_power

def get_optimization_technique_idx(value, technique):
    if technique == 'channel':
        idx = np.where(acc_c[0,:] == value)[0][0]

    elif technique == 'quantization':
        idx = np.where(acc_q[0,:] == value)[0][0]

    elif technique == 'pruning':
        idx = np.where(acc_p[0,:] == value)[0][0]
    return idx

def greedy_search(best_c, best_q, best_p, best_acc_actual, best_nparam):
    if args.printprocess:
        print(best_p, best_q, best_c, best_acc_actual)

    best_c_list = [best_c]
    best_q_list = [best_q]
    best_p_list = [best_p]
    best_acc_actual_list = [best_acc_actual]
    best_nparam_list = [best_nparam]
    acc_diff_list = [0]

    idx_c_list = []
    idx_q_list = []
    idx_p_list = []

    idx_c = get_optimization_technique_idx(best_c, 'channel')
    idx_q = get_optimization_technique_idx(best_q, 'quantization')
    idx_p = get_optimization_technique_idx(best_p, 'pruning')

    if idx_c == 0:
        idx_c_list.append(acc_c[0,1])
    elif (idx_c+1) == acc_c.shape[1]:
        idx_c_list.append(acc_c[0,-2])
    else:
        idx_c_list.append(acc_c[0,idx_c-1])
        idx_c_list.append(acc_c[0,idx_c+1])

    if idx_q == 0:
        idx_q_list.append(acc_q[0,1])
    elif (idx_q+1) == acc_q.shape[1]:
        idx_q_list.append(acc_q[0,-2])
    else:
        idx_q_list.append(acc_q[0,idx_q-1])
        idx_q_list.append(acc_q[0,idx_q+1])

    if idx_p == 0:
        idx_p_list.append(acc_p[0,1])
    elif (idx_p+1) == acc_p.shape[1]:
        idx_p_list.append(acc_p[0,-2])
    else:
        idx_p_list.append(acc_p[0,idx_p-1])
        idx_p_list.append(acc_p[0,idx_p+1])

    idx_c_array = np.array(idx_c_list)
    idx_q_array = np.array(idx_q_list)
    idx_p_array = np.array(idx_p_list)

    if args.pparam != 0:
        for best_c_greedy in idx_c_array:
            nparam = nparam_net * pow(best_c_greedy,2) * best_q/32 * (100-best_p)/100
            if (args.pparam/100*nparam_net >= nparam):
                best_c_list.append(best_c_greedy)
                best_q_list.append(best_q)
                best_p_list.append(best_p)
                best_acc_actual_list.append(get_actual_acc(best_c_greedy, best_q, best_p))
                best_nparam_list.append(nparam)
                acc_diff_list.append(get_actual_acc(best_c_greedy, best_q, best_p) - best_acc_actual)
            else:
                pass

        for best_q_greedy in idx_q_array:
            nparam = nparam_net * pow(best_c,2) * best_q_greedy/32 * (100-best_p)/100
            if (args.pparam/100*nparam_net >= nparam):
                best_c_list.append(best_c)
                best_q_list.append(best_q_greedy)
                best_p_list.append(best_p)
                best_nparam_list.append(nparam)
                best_acc_actual_list.append(get_actual_acc(best_c, best_q_greedy, best_p))
                acc_diff_list.append(get_actual_acc(best_c, best_q_greedy, best_p) - best_acc_actual)
            else:
                pass

        for best_p_greedy in idx_p_array:
            nparam = nparam_net * pow(best_c,2) * best_q/32 * (100-best_p_greedy)/100
            if (args.pparam/100*nparam_net >= nparam):
                best_c_list.append(best_c)
                best_q_list.append(best_q)
                best_p_list.append(best_p_greedy)
                best_nparam_list.append(nparam)
                best_acc_actual_list.append(get_actual_acc(best_c, best_q, best_p_greedy))
                acc_diff_list.append(get_actual_acc(best_c, best_q, best_p_greedy) - best_acc_actual)
            else:
                pass

        idx = np.where(best_acc_actual_list == np.max(best_acc_actual_list))[0][0]
        best_c = best_c_list[idx]
        best_q = best_q_list[idx]
        best_p = best_p_list[idx]
        best_nparam = best_nparam_list[idx]
        best_acc_actual = best_acc_actual_list[idx]

    else:
        for best_c_greedy in idx_c_array:
            nparam = nparam_net * pow(best_c_greedy,2) * best_q/32 * (100-best_p)/100
            acc = get_actual_acc(best_c_greedy, best_q, best_p)
            if (net_acc - args.accdrop <= acc) and (best_nparam >= nparam):
                best_c_list.append(best_c_greedy)
                best_q_list.append(best_q)
                best_p_list.append(best_p)
                best_nparam_list.append(nparam)
                best_acc_actual_list.append(acc)
                acc_diff_list.append(get_actual_acc(best_c_greedy, best_q, best_p) - best_acc_actual)
            else:
                pass

        for best_q_greedy in idx_q_array:
            nparam = nparam_net * pow(best_c,2) * best_q_greedy/32 * (100-best_p)/100
            acc = get_actual_acc(best_c, best_q_greedy, best_p)
            if (net_acc - args.accdrop <= acc) and (best_nparam >= nparam):
                best_c_list.append(best_c)
                best_q_list.append(best_q_greedy)
                best_p_list.append(best_p)
                best_nparam_list.append(nparam)
                best_acc_actual_list.append(acc)
                acc_diff_list.append(get_actual_acc(best_c, best_q_greedy, best_p) - best_acc_actual)
            else:
                pass

        for best_p_greedy in idx_p_array:
            nparam = nparam_net * pow(best_c,2) * best_q/32 * (100-best_p_greedy)/100
            acc = get_actual_acc(best_c, best_q, best_p_greedy)
            if (net_acc - args.accdrop <= acc) and (best_nparam >= nparam):
                best_c_list.append(best_c)
                best_q_list.append(best_q)
                best_p_list.append(best_p_greedy)
                best_nparam_list.append(nparam)
                best_acc_actual_list.append(acc)
                acc_diff_list.append(get_actual_acc(best_c, best_q, best_p_greedy) - best_acc_actual)
            else:
                pass

        idx = np.where(best_nparam_list == np.min(best_nparam_list))[0][0]
        best_c = best_c_list[idx]
        best_q = best_q_list[idx]
        best_p = best_p_list[idx]
        best_nparam = best_nparam_list[idx]
        best_acc_actual = best_acc_actual_list[idx]

    if args.printprocess:
        print(best_p_list)
        print(best_q_list)
        print(best_c_list)
        print(best_nparam_list)
        print(best_acc_actual_list)
        print(acc_diff_list)

    #print(np.where(acc_diff_list == np.max(acc_diff_list))[0][0])


    if args.printprocess:
        print('=============================================')
    return best_c, best_q, best_p, best_acc_actual, best_nparam

def get_grad(c1, q1, p1, c2, q2, p2):
    acc1 = get_actual_acc(c1, q1, p1)
    acc2 = get_actual_acc(c2, q2, p2)
    nparam1 = nparam_net * (100-p1)/100 * q1/32 * pow(c1,2)
    nparam2 = nparam_net * (100-p2)/100 * q2/32 * pow(c2,2)
    grad = (acc1-acc2)/(nparam1-nparam2)
    return grad

#def grad_based_search(best_c, best_q, best_p, power_c, power_q, power_p, best_acc_actual, best_nparam):
#    if args.printprocess:
#        print(best_p, best_q, best_c, best_acc_actual)
#
#    best_c_list = [best_c]
#    best_q_list = [best_q]
#    best_p_list = [best_p]
#    best_acc_actual_list = [best_acc_actual]
#    best_nparam_list = [best_nparam]
#    acc_diff_list = [0]
#
#    idx_c_list = []
#    idx_q_list = []
#    idx_p_list = []
#
#    idx_c = get_optimization_technique_idx(best_c, 'channel')
#    idx_q = get_optimization_technique_idx(best_q, 'quantization')
#    idx_p = get_optimization_technique_idx(best_p, 'pruning')
#
#    if idx_c == 0:
#        idx_c_list.append(acc_c[0,1])
#        idx_c_list.append(acc_c[0,2])
#    elif idx_c == 1:
#        idx_c_list.append(acc_c[0,0])
#        idx_c_list.append(acc_c[0,2])
#        idx_c_list.append(acc_c[0,3])
#    elif (idx_c+2) == acc_c.shape[1]:
#        idx_c_list.append(acc_c[0,-4])
#        idx_c_list.append(acc_c[0,-3])
#        idx_c_list.append(acc_c[0,-1])
#    elif (idx_c+1) == acc_c.shape[1]:
#        idx_c_list.append(acc_c[0,-3])
#        idx_c_list.append(acc_c[0,-2])
#    else:
#        idx_c_list.append(acc_c[0,idx_c-2])
#        idx_c_list.append(acc_c[0,idx_c-1])
#        idx_c_list.append(acc_c[0,idx_c+1])
#        idx_c_list.append(acc_c[0,idx_c+2])
#
#    if idx_q == 0:
#        idx_q_list.append(acc_q[0,1])
#        idx_q_list.append(acc_q[0,2])
#    elif idx_q == 1:
#        idx_q_list.append(acc_q[0,0])
#        idx_q_list.append(acc_q[0,2])
#        idx_q_list.append(acc_q[0,3])
#    elif (idx_q+2) == acc_q.shape[1]:
#        idx_q_list.append(acc_q[0,-4])
#        idx_q_list.append(acc_q[0,-3])
#        idx_q_list.append(acc_q[0,-1])
#    elif (idx_q+1) == acc_q.shape[1]:
#        idx_q_list.append(acc_q[0,-3])
#        idx_q_list.append(acc_q[0,-2])
#    else:
#        idx_q_list.append(acc_q[0,idx_q-2])
#        idx_q_list.append(acc_q[0,idx_q-1])
#        idx_q_list.append(acc_q[0,idx_q+1])
#        idx_q_list.append(acc_q[0,idx_q+2])
#
#    if idx_p == 0:
#        idx_p_list.append(acc_p[0,1])
#        idx_p_list.append(acc_p[0,2])
#    elif idx_p == 1:
#        idx_p_list.append(acc_p[0,0])
#        idx_p_list.append(acc_p[0,2])
#        idx_p_list.append(acc_p[0,3])
#    elif (idx_p+2) == acc_p.shape[1]:
#        idx_p_list.append(acc_p[0,-4])
#        idx_p_list.append(acc_p[0,-3])
#        idx_p_list.append(acc_p[0,-1])
#    elif (idx_p+1) == acc_p.shape[1]:
#        idx_p_list.append(acc_p[0,-3])
#        idx_p_list.append(acc_p[0,-2])
#    else:
#        idx_p_list.append(acc_p[0,idx_p-2])
#        idx_p_list.append(acc_p[0,idx_p-1])
#        idx_p_list.append(acc_p[0,idx_p+1])
#        idx_p_list.append(acc_p[0,idx_p+2])
#
#    idx_c_array = np.array(idx_c_list)
#    idx_q_array = np.array(idx_q_list)
#    idx_p_array = np.array(idx_p_list)
#    
#    print(idx_c_array)
#    print(idx_q_array)
#    print(idx_p_array)
#
#    
#    for point_c in idx_c_array:
#        for point_q in idx_q_array:
#            for point_p in idx_p_array:
#                idx_c = np.where(acc_c[0,:] == point_c)[0][0]
#                idx_p = np.where(acc_p[0,:] == point_p)[0][0]
#                idx_q = np.where(acc_q[0,:] == point_q)[0][0]
#                acc = net_acc * pow(acc_c[1,idx_c],power_c) * pow(acc_q[1,idx_q],power_q) * pow(acc_p[1,idx_p],power_p)
#                nparam = nparam_net * (100-point_p)/100 * point_q/32 * pow(point_c,2)
#                grad = (best_acc_actual - 
#                
#    exit()

def acc_based_search():
    power_c = 1
    power_q = 1
    power_p = 1

    counter = 0
    target_acc = net_acc - args.accdrop
    if args.printprocess:
        print('target acc : ',target_acc)
    best_c_list = []
    best_q_list = []
    best_p_list = []
    best_acc_list = []

    while 1:
        global performance_estimated
        performance_estimated = []
        best_c, best_q, best_p, best_nparam, best_acc_est = \
                get_best_combination_acc_based(power_c ,power_q, power_p)
        best_acc_actual = get_actual_acc(best_c, best_q, best_p)
        flag_acc = net_acc - args.accdrop <= best_acc_actual
        flag_error = abs(get_actual_acc(best_c, best_q, best_p) - best_acc_est) < args.thres

        if flag_acc and flag_error :
            break
        elif counter > 15:
            best_c_list = np.array(best_c_list)
            best_p_list = np.array(best_p_list)
            best_q_list = np.array(best_q_list)
            best_acc_list = np.array(best_acc_list)

            acc_diff_list = best_acc_list - target_acc
            idx = np.where(abs(acc_diff_list) == np.min(abs(acc_diff_list)))
            #idx = np.where(np.min(acc_diff_list))

            if len(idx[0]) == 0:
                idx = [[0]]
            if args.printprocess:
                print('best acc est list : ',best_acc_list)
            
            best_c = best_c_list[idx[0][0]]
            best_p = best_p_list[idx[0][0]]
            best_q = best_q_list[idx[0][0]]
            best_acc_list = best_acc_list[idx[0][0]]
            best_acc_actual = get_actual_acc(best_c, best_q, best_p)
            best_nparam = nparam_net * (100-best_p)/100 * best_q/32 * pow(best_c,2)
            #best_nparam = 100000000
            break
        elif counter > 5:
            best_c_list.append(best_c)
            best_p_list.append(best_p)
            best_q_list.append(best_q)
            best_acc_list.append(get_actual_acc(best_c, best_q, best_p))
            counter += 1

            #print("counter = ",counter)
            #print(best_p)
            #print(best_q)
            #print(best_c)
            #print(get_actual_acc(best_c, best_q, best_p))
            #print(best_acc)
            #print(hardware_metric)
            #print("============================")

            if args.unified:
                power_c, power_q, power_p = modify_power_factors_unified(best_c, best_q, best_p)
            else:
                power_c, power_q, power_p = modify_power_factors(best_c, best_q, best_p)
        else:
            if args.unified:
                power_c, power_q, power_p = modify_power_factors_unified(best_c, best_q, best_p)
            else:
                power_c, power_q, power_p = modify_power_factors(best_c, best_q, best_p)
            best_acc_actual = get_actual_acc(best_c, best_q, best_p)
            best_nparam = nparam_net * (100-best_p)/100 * best_q/32 * pow(best_c,2)
            counter += 1


    for i in range(args.ngreedy):
            
        best_c_prev = best_c
        best_q_prev = best_q
        best_p_prev = best_p

        if args.printprocess:
            print("search "+str(i)+", best_p, best_q, best_c, best_acc_actual, best_nparam : "\
                    +str(best_p), str(best_q), str(best_c), str(best_acc_actual), str(best_nparam))

        best_c, best_q, best_p, best_acc_actual, best_nparam = \
                greedy_search(best_c, best_q, best_p, get_actual_acc(best_c, best_q, best_p), best_nparam)

        if (best_c_prev == best_c) and (best_q_prev == best_q) and (best_p_prev == best_p):
            break


    #if args.netsel == 1:
    #    print('network :\t\t\t\tResNet18')
    #elif args.netsel == 2:
    #    print('network :\t\t\t\tSqueezeNext')
    #elif args.netsel == 3:
    #    print('network :\t\t\t\tMobileNetV2')
    #else:
    #    print('network :\t\t\t\tVGG16')

    #print('Size constraint :\t\t\t{}'.format(args.pparam))
    #print('Pruning factor :\t\t\t{}'.format(best_p))
    #print('Quantization factor :\t\t\t{}'.format(best_q))
    #print('Channel scaling factor :\t\t{}'.format(best_c))
    #print('Actual accuracy :\t\t\t{:.2f}'.format(get_actual_acc(best_c, best_q, best_p)))
    #print('Estimated accuracy :\t\t\t{:.2f}'.format(best_acc))
    #print('Hardware metric :\t\t\t{:.2f}'.format(hardware_metric))
    #print('Minimum number of nonzero param :\t{:.2f}'.format(best_nparam))

    ########### printing predicted optimal combination information ###############
    if args.actual == 0:
        print(best_p)
        print(best_q)
        print(best_c)
        print(get_actual_acc(best_c, best_q, best_p))
        print(best_acc_est)
        if best_nparam == 100000000:
            best_nparam = nparam_net * (100-best_p)/100 * best_q/32 * pow(best_c,2)
        print(best_nparam)

    ########### printing actual optimal combination information ###############
    if args.actual == 1:
        print(best_p)
        print(best_q)
        print(best_c)
        print(best_acc_est)
        print(best_nparam)

if __name__ == '__main__':
    acc_based_search()
