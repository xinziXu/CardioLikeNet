import numpy as np
import torch
import os
# from modules_npy import conv1d_npy, bn1d_npy, relu_npy, conv1dtrans_npy,lstm_npy,conv1d_npy_quan,conv1dtrans_npy_quan,lstm_npy_quan,bn1d_npy_quan,bit, cnn_bn_fusion

bit = 10
def Quant(input, scale, zero_point):

    x_q = torch.clamp(torch.round(input/scale + zero_point),0,128 )

    return x_q

def DeQuant(input, scale, zero_point):

    # print('scale',scale)
    # print('zp', zero_point)
    # input = input.int()
    # print(input[0,0,0])
    # print(input[0,0,0] - zero_point)
    x_dq = scale * (input - zero_point)
    # print(x_dq[0,0,0])
    return x_dq

def quan_xbit (input, bit = bit):
    output = torch.floor(input*(2**bit))
    return output

def Linear_Relu(input, W, bias ):
    '''
    input: [batch_size, in_features]
    W: [out_features, in_features]
    bias: [out_features]
    output: [batch_size, out_features]
    '''
    batch_size = input.shape[0]
    out_features = W.shape[0]
    bias_expand = bias.unsqueeze(0)
    bias_expand.repeat(batch_size,out_features)
    output = torch.matmul(input, W.T) + bias_expand
    zero = torch.zeros_like(output)
    output = torch.max(output, zero)
    return output

def Linear_Relu_quan(input, W, bias ):
    '''
    input: [batch_size, in_features]
    W: [out_features, in_features]
    bias: [out_features]
    output: [batch_size, out_features]
    '''
    # input = quan_xbit(input)
    W = quan_xbit(W)
    bias = quan_xbit(bias)


    batch_size = input.shape[0]
    out_features = W.shape[0]
    bias_expand = bias.unsqueeze(0)
    bias_expand.repeat(batch_size,out_features)
    output = torch.floor(torch.matmul(input, W.T)/(2**bit)) + bias_expand
    zero = torch.zeros_like(output)
    output = torch.max(output, zero)
    return output

ckpt_path = '/home/xuxinzi/ECG_CLF/code/code_matlab/ARR_CLF/ckpt/ann_model_best.pth'

def network_forward(input):
    ckpt = torch.load(ckpt_path)
    state = ckpt['model_state']
    # for key in state.keys():
    #     print(key)



    input = quan_xbit(input)
    fc1 = Linear_Relu_quan(input, state['ann.linear1.weight'], state['ann.linear1.bias'])
    fc2 = Linear_Relu_quan(fc1, state['ann.linear2.weight'], state['ann.linear2.bias'])

    route = './data_torch/' + 'fc2_self_quan.txt'
    tmp = fc2[0].detach().cpu().numpy()
    with open(route, 'a') as txt:
        np.savetxt(txt, tmp, delimiter='\n')     
    return fc2




