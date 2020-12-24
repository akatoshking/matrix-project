import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class tt_conv_full(nn.Module):
    def __init__(self,
                    window, 
                    inp_ch_modes, 
                    out_ch_modes, \
                    ranks, \
                    strides=[1, 1],\
                    padding='SAME',\
                    filters_initializer=torch.nn.init.xavier_uniform_,\
                    cores_initializer=torch.nn.init.xavier_uniform_,\
                    biases_initializer=torch.nn.init.zeros_
                    ):
        """ tt-conv-layer (convolution of full input tensor with tt-filters (make tt full then use conv2d))
        Args:
            inp: input tensor, float - [batch_size, H, W, C]
            window: convolution window size, list [wH, wW]
            inp_ch_modes: input channels modes, np.array (int32) of size d
            out_ch_modes: output channels modes, np.array (int32) of size d
            ranks: tt-filters ranks, np.array (int32) of size (d + 1)        
            strides: strides, list of 2 ints - [sx, sy] 
            padding: 'SAME' or 'VALID', string
            filters_initializer: filters init function
            filters_regularizer: filters regularizer function
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
        super(tt_conv_full, self).__init__()
        self.ranks = ranks 
        self.out_modes = out_ch_modes 
        self.in_modes = inp_ch_modes
        self.window = window
        self.strides = strides 
        if padding=='SAME' and strides==[1,1]:
            self.padding = ((window[0]-1)//2,(window[0]-1)//2)#stride=1
            # print(type(self.padding))
        elif padding=='VALID':
            self.padding = 0
        elif padding==1:
            self.padding = 1
        # self.padding = padding 
        
        filter_shape = [window[0], window[1], ranks[0]]
        if (window[0] * window[1] * 1 * ranks[0] == 1):
            self.filters = torch.empty(*filter_shape, device=device)
            torch.nn.init.ones_(self.filters)
            self.filters = Parameter(self.filters, requires_grad=True)
        else:
            self.filters = torch.empty(*filter_shape, device=device)
            filters_initializer(self.filters)
            self.filters = Parameter(self.filters, requires_grad=True)
        self.dim = inp_ch_modes.size
        
        self.mat_cores = []
        for i in range(self.dim):
            if type(cores_initializer) == list:
                cinit = cores_initializer[i]
            else:
                cinit = cores_initializer
            w = torch.empty(out_ch_modes[i] * ranks[i + 1], ranks[i] * inp_ch_modes[i], device=device)
            cinit(w)
            w = Parameter(w, requires_grad=True)
            self.mat_cores.append(w)
        
        self.fshape = [window[0], window[1]]
        self.order = [0, 1]
        inord = []
        outord = []
        for i in range(self.dim):
            self.fshape.append(inp_ch_modes[i])
            inord.append(2 + 2 * i)
            self.fshape.append(out_ch_modes[i])
            outord.append(2 + 2 * i + 1)
        self.order += inord + outord

        if biases_initializer is not None:
            self.biases = torch.empty(np.prod(out_ch_modes), device=device, requires_grad=True)
            biases_initializer(self.biases)
            self.biases = Parameter(self.biases, requires_grad=True)
        else:
            self.biases = torch.empty(np.prod(out_ch_modes), device=device, requires_grad=True)
            torch.nn.init.zeros_(self.biases)
            self.biases = Parameter(self.biases, requires_grad=False)
        # bs,ch,h,w = x.shape
        # out = torch.reshape(x, [-1,h,w,ch])

        

    def forward(self, x):
        # print(self.padding)
        full = self.filters
        # print('filters',self.filters.shape)
        for i in range(self.dim):
            full = torch.reshape(full, [-1, self.ranks[i]])
            core = torch.transpose(self.mat_cores[i],0,1)
            core = torch.reshape(core, [self.ranks[i], -1])
            full = torch.matmul(full, core)
        full = torch.reshape(full, self.fshape)
        full = full.permute( *self.order)
        full = torch.reshape(full, [np.prod(self.out_modes), np.prod(self.in_modes), self.window[0], self.window[1] ])
        # full = torch.reshape(full, [self.window[0], self.window[1], np.prod(self.in_modes), np.prod(self.out_modes)])#[filter_height, filter_width, in_channels, out_channels]
        # full = full.permute(3,2,0,1)#out_channels, in_channels,H,W
        # full = Parameter(full)
        # print(type(full))
        out = F.conv2d(input=x, weight=full, bias=self.biases, stride=self.strides, padding=self.padding)
        return out

# batch_in = torch.randn(100,64,128,128)

# model = tt_conv_full([3, 3],np.array([4,4,4],dtype=np.int32),np.array([4,4,4],dtype=np.int32),np.array([16,16,16,1],dtype=np.int32))
# model = model.to(device=device)
# output = model(batch_in)
# print(output.shape)

