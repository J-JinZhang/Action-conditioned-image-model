import torch.nn as nn
import torch.nn.functional as F
import torch
from math import sqrt



class PixelwiseNorm(nn.Module):
    def __init__(self):
        super(PixelwiseNorm, self).__init__()

    def forward(self, x, alpha=1e-8):
        #y = x.pow(2.).mean(dim=1, keepdim=True).add(alpha).sqrt()  # [N1HW]
        #y = x / y  # normalize the input x volume
        #return y
        return x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)

    

class equalized_conv2d(nn.Module):
    def __init__(self, c_in, c_out, k_size, stride=1, pad=0, bias=True):
        from torch.nn.modules.utils import _pair
        from numpy import sqrt, prod
        super().__init__()
        # define the weight and bias if to be used
        self.weight = nn.Parameter( nn.init.normal_(torch.rand(c_out, c_in, *_pair(k_size))) )
        self.use_bias = bias
        self.stride = stride
        self.pad = pad
        if self.use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(c_out).fill_(0))
        fan_in = prod(_pair(k_size)) * c_in  # value of fan_in
        self.scale = sqrt(2) / sqrt(fan_in)

    def forward(self, x):
        from torch.nn.functional import conv2d
        return conv2d(input=x,
                      weight=self.weight * self.scale,  # scale the weight on runtime
                      bias=self.bias if self.use_bias else None,
                      stride=self.stride,
                      padding=self.pad)

    def extra_repr(self):
        return ", ".join(map(str, self.weight.shape))


    
class equalized_deconv2d(nn.Module):
    def __init__(self, c_in, c_out, k_size, stride=1, pad=0, bias=True):
        from torch.nn.modules.utils import _pair
        from numpy import sqrt
        super().__init__()
        # define the weight and bias if to be used
        self.weight = nn.Parameter( nn.init.normal_(torch.rand(c_in, c_out, *_pair(k_size))) )
        self.use_bias = bias
        self.stride = stride
        self.pad = pad
        if self.use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(c_out).fill_(0))
        fan_in = c_in  # value of fan_in for deconv
        self.scale = sqrt(2) / sqrt(fan_in)

    def forward(self, x):
        from torch.nn.functional import conv_transpose2d
        return conv_transpose2d(input=x,
                                weight=self.weight * self.scale,  # scale the weight on runtime
                                bias=self.bias if self.use_bias else None,
                                stride=self.stride,
                                padding=self.pad)

    def extra_repr(self):
        return ", ".join(map(str, self.weight.shape))


    
class equalized_linear(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        from numpy import sqrt
        super().__init__()
        self.weight = nn.Parameter( nn.init.normal_(torch.rand(c_out, c_in)) )
        self.use_bias = bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(c_out).fill_(0))
        fan_in = c_in
        self.scale = sqrt(2) / sqrt(fan_in)

    def forward(self, x):
        from torch.nn.functional import linear
        return linear(x, self.weight * self.scale,
                      self.bias if self.use_bias else None)

    
