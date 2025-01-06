import numpy as np
from math import floor, ceil
class Conv2d():
  def __init__(self, 
               kernel_size = 3,
               input_channel = 3, 
               output_channel = 64,
               stride = 1, 
               padding = 0):

    assert(kernel_size >= 0 and "Kernel size must be positive")
    assert(input_channel >= 0 and "Input channel must be positive")
    assert(output_channel >= 0 and "Output channel must be positive")
    assert(stride > 0 and 'Stride must be positive')
    assert((padding == 0 or padding == 1) and 'Padding must be 0 | 1')

    self.kwargs = {
    'kernel_size' : kernel_size,
    'input_channel' : input_channel,
    'output_channel' : output_channel,
    'stride' : stride,
    'padding' : padding
    }

    self.filter = np.random.normal(loc=.0, scale=.4,
        size=(output_channel, kernel_size, kernel_size))
    self.bias = np.random.normal(loc=.0, scale=.4,
        size=(output_channel))
    
  def pad(self, x):
    self.input_dimensions = {
      'B':x.shape[0],
      'C':x.shape[1],
      'W':x.shape[2],
      'H':x.shape[3],
    }

    B, C, W, H = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    S, K = self.kwargs['stride'], self.kwargs['kernel_size']
    padding = ceil(((W - 1) * S + 1 - H + (K - 1)) / 2)
    P_W, P_H = W + 2 * padding, H + 2 * padding
    return np.pad(
            x.reshape(B * C, W, H), padding)[padding:-padding].reshape(B, C, P_W, P_H)

  def __get_filter_top_left_corner(self, x):
    S, K = self.kwargs['stride'], self.kwargs['kernel_size']

    __kernel_tl_points = np.arange(0, x.shape[-1] - K + S, S)
    _conv_coords = np.stack([np.repeat(__kernel_tl_points, len(__kernel_tl_points), axis=0), 
                 np.repeat(__kernel_tl_points.reshape(1, -1), len(__kernel_tl_points),
                 axis=0).flatten()], axis=1)
    
    return _conv_coords
    
  def forward(self, X):
    X = self.pad(X) if self.kwargs['padding'] else X # padded
    S, K = self.kwargs['stride'], self.kwargs['kernel_size']

    _conv_coords = self.__get_filter_top_left_corner(X) 
    __conv_range_x = _conv_coords[:, 0][:, np.newaxis] + np.arange(K, dtype=np.int8) 
    __conv_range_y = _conv_coords[:, 1][:, np.newaxis] + np.arange(K, dtype=np.int8) 

    X = X[:, :, __conv_range_x[:, :, np.newaxis], __conv_range_y[:, np.newaxis, :]]
    
  def backward(self, loss):
    pass

  def call(self, x):
    pass
  

l1 = Conv2d(kernel_size=3, padding=0, stride=2)
l1.forward(np.arange(30 * 7 ** 2).reshape(10,3,7,7))
