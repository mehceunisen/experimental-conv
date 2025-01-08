import numpy as np
from math import floor, ceil
from typing import Tuple
class Conv2d():
  def __init__(self, 
               kernel_size: Tuple[int, int]=(3, 3),
               input_channel = 3, 
               output_channel = 64,
               stride = 1, 
               padding = 0):
    assert(len(kernel_size) == 2 and "Kernel size must be (N X M)")
    assert(input_channel >= 0 and "Input channel must be positive")
    assert(output_channel >= 0 and "Output channel must be positive")
    assert(stride > 0 and ')Stride must be positive')
    assert((padding == 0 or padding == 1) and 'Padding must be 0 | 1')

    self.kwargs = {
      'kernel_size' : kernel_size,
      'input_channel' : input_channel,
      'output_channel' : output_channel,
      'stride' : stride,
      'padding' : padding
    }

    self.filters = np.random.normal(loc=.0, scale=.4,
        size=(output_channel, kernel_size[0], kernel_size[1]))
    self.bias = np.random.normal(loc=.0, scale=.4,
        size=(output_channel))
    
  def __get_filter_top_left_corner(self, x):
    # needs attention
    S, K_A, K_B = self.kwargs['stride'], *self.kwargs['kernel_size']
    a = __kernel_top_left_points_a = np.arange(0, x.shape[-2] - K_A + 1, S)
    b = __kernel_top_left_points_b = np.arange(0, x.shape[-1] - K_B + 1, S)
    
    _a = np.repeat(a[:, np.newaxis], b.shape[0], axis=1)
    _b = np.repeat(b[:, np.newaxis], a.shape[0], axis=1).T

    return np.stack((_a, _b), axis=2).reshape(-1, 2)
    

  def __pad(self, x):
    B, C, W, H = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    S, K_A, K_B = self.kwargs['stride'], *self.kwargs['kernel_size']
    pad_a_x = pad_a_y = ((W - 1) * S + 1 - H + (K_A - 1)) / 2
    pad_b_x = pad_b_y = ((W - 1) * S + 1 - H + (K_B - 1)) / 2
     
    if pad_a_x % 1 != 0:
        pad_a_x = int(pad_a_x) + 1
    if pad_b_x % 1 != 0:
        pad_b_x = int(pad_b_x) + 1

    pad_a_x = int(pad_a_x)
    pad_a_y = int(pad_a_y)

    pad_b_x = int(pad_b_x)
    pad_b_y = int(pad_b_y)
    
    return (np.pad(x, ((0,0), (0,0), (pad_a_x, pad_a_y), (pad_b_x, pad_b_y))), 
            int((pad_a_x + pad_a_y) / 2), int((pad_b_x + pad_b_y) / 2))

  def __create_patches(self, X):
    K_A, K_B = self.kwargs['kernel_size'][:]
    _conv_coords = self.__get_filter_top_left_corner(X)
    _x = _conv_range_x = _conv_coords[:, 0][:, np.newaxis] + np.arange(K_A, dtype=np.int8) 
    _y = _conv_range_y = _conv_coords[:, 1][:, np.newaxis] + np.arange(K_B, dtype=np.int8) 
    X = X[:, :, _conv_range_x[:, :, np.newaxis], _conv_range_y[:, np.newaxis, :]]
    return X
  
  def forward(self, X):

    assert(len(X.shape) == 4 and "Input dimensions should be as (B, C, W, H)")

    B, C, W, H = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
    O_C = self.kwargs['output_channel']
    S, K_A, K_B = self.kwargs['stride'], *self.kwargs['kernel_size']
    assert((K_A <= W or K_B <= W) and "Kernel size must be smaller than input (H,W)")

    X, padding_size_a, padding_size_b = (self.__pad(X) 
                if self.kwargs['padding'] else (X, 0, 0)) # padded
    
    _out_dim_a = _out_dim_b = H

    if padding_size_a == 0:
      _out_dim_a = int((H + padding_size_a - K_A) / S) + 1
    if padding_size_b == 0:
      _out_dim_b = int((H + padding_size_b - K_B) / S) + 1
    X = self.__create_patches(X)

    X = (np.sum(X[:, :, :, np.newaxis] * self.filters, axis=(1, -1, -2))
    .transpose((0, 2, 1)).reshape(B, O_C, _out_dim_a, _out_dim_b))
    + self.bias[:, np.newaxis, np.newaxis]

    return X

  def backward(self, loss):
    pass

  def call(self, x):
    pass
  
l1 = Conv2d(kernel_size=(4,3), padding=0, stride=7, output_channel=6)
print(l1.forward(np.arange(3 * 5 ** 2).reshape(1,3,5,5)).shape)
