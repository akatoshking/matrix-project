import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class tt_fc(nn.Module):
    def __init__(self,inp_modes,out_modes,mat_ranks,cores_initializer=torch.nn.init.xavier_uniform_,biases_initializer= torch.nn.init.zeros_):
    """ tt-layer (tt-matrix by full tensor product)
    Args:
        inp: input tensor, float - [batch_size, prod(inp_modes)]
        inp_modes: input tensor modes, i.e., The shape of input tensor?
        out_modes: output tensor modes, i.e., The shape of output tensor?
        mat_ranks: tt-matrix ranks
        cores_initializer: cores init function, could be a list of functions for specifying different function for each core
        cores_regularizer: cores regularizer function, could be a list of functions for specifying different function for each core
        biases_initializer: biases init function (if None then no biases will be used)
        biases_regularizer: biases regularizer function        
        trainable: trainable variables flag, bool
        cpu_variables: cpu variables flag, bool
        scope: layer variable scope name, string
    Returns:
        out: output tensor, float - [batch_size, prod(out_modes)]
    """
        super(tt_fc, self).__init__()
        self.weights = []
        self.dim = inp_modes.size
        self.mat_cores = []
        self.in_dims = np.prod(inp_modes)
        self.mat_ranks = mat_ranks
        self.inp_modes = inp_modes
        self.out_modes = out_modes
        for i in range(self.dim):
            if type(cores_initializer) == list:
                cinit = cores_initializer[i]
            else:
                cinit = cores_initializer
            
            # if type(cores_regularizer) == list:
            #     creg = cores_regularizer[i]
            # else:
            #     creg = cores_regularizer
            w = torch.empty(out_modes[i] * mat_ranks[i + 1], mat_ranks[i] * inp_modes[i], device=device)
            cinit(w)
            self.mat_cores.append(w)
        if biases_initializer is not None:
            self.biases = torch.empty(np.prod(out_modes), device=device)
            biases_initializer(self.biases)
        else:
            self.biases = torch.empty(np.prod(out_modes), device=device)
            torch.nn.init.zeros_(self.biases)

    def forward(self, x):
        out = torch.reshape(x, [-1, self.in_dims])
        out = torch.transpose(out,0,1)
        for i in range(self.dim):
            out = torch.reshape(out, [self.mat_ranks[i] * self.inp_modes[i], -1])            
            out = torch.matmul(self.mat_cores[i], out)
            out = torch.reshape(out, [self.out_modes[i], -1])
            out = torch.transpose(out, 0, 1)
        out = torch.reshape(out,[-1,np.prod(self.out_modes)])
        out = torch.add(out, self.biases)
        return out

# batch_in = torch.randn(64,3072)
# # opts['inp_modes_1'] = np.array([4, 4, 4, 4, 4, 3], dtype='int32')
# # opts['out_modes_1'] = np.array([8, 8, 8, 8, 8, 8], dtype='int32')
# model = tt_fc(np.array([4, 4, 4, 4, 4, 3], dtype='int32'),np.array([8, 8, 8, 8, 8, 8], dtype='int32'),np.array([1, 3, 3, 3, 3, 3, 1], dtype='int32'))
# model = model.to(device=device)
# output = model(batch_in)
# print(output.shape)