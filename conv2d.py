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
        size=(output_channel, kernel_size, kernel_size))
    self.filters = np.ones(output_channel * kernel_size ** 2).reshape(output_channel, kernel_size, kernel_size)
    self.bias = np.random.normal(loc=.0, scale=.4,
        size=(output_channel))
    
  def __pad(self, x):
    B, C, W, H = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    S, K = self.kwargs['stride'], self.kwargs['kernel_size']
    
    pad_a = pad_b = ((W - 1) * S + 1 - H + (K - 1)) / 2
     
    if pad_a % 1 != 0:
        pad_a = int(pad_a) + 1
    pad_a = int(pad_a)
    pad_b = int(pad_b)

    return np.pad(x, ((0,0), (0,0), (pad_a, pad_b), (pad_a, pad_b))), pad_a + pad_b

  def __get_filter_top_left_corner(self, x):
    S, K = self.kwargs['stride'], self.kwargs['kernel_size']
    __kernel_tl_points = np.arange(0, x.shape[-1] - K + 1, S)
    _conv_coords = np.stack([
        np.repeat(__kernel_tl_points, len(__kernel_tl_points), axis=0), 
        np.repeat(__kernel_tl_points.reshape(1, -1), len(__kernel_tl_points),
        axis=0).flatten()], axis=1)
    
    return _conv_coords

  def __create_patches(self, X):
    K = self.kwargs['kernel_size']
    _conv_coords = self.__get_filter_top_left_corner(X) 
    _conv_range_x = _conv_coords[:, 0][:, np.newaxis] + np.arange(K, dtype=np.int8) 
    _conv_range_y = _conv_coords[:, 1][:, np.newaxis] + np.arange(K, dtype=np.int8) 

    X = X[:, :, _conv_range_x[:, :, np.newaxis], _conv_range_y[:, np.newaxis, :]]
    return X
  #assumption, input and the kernel is square (H = W), might fix it later
  def forward(self, X):
    assert(len(X.shape) == 4 and "Input dimensions should be as (B, C, W, H)")

    B, C, W, H = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
    O_C = self.kwargs['output_channel']
    S, K = self.kwargs['stride'], self.kwargs['kernel_size']
    
    assert(K <= W and "Kernel size must be smaller than input (H,W)")

    X, padding_size = self.__pad(X) if self.kwargs['padding'] else (X, 0) # padded
    _out_dim = int((H + padding_size - K) / S) + 1 if padding_size == 0 else H 
    X = self.__create_patches(X)

    X = (np.sum(X[:, :, :, np.newaxis] * self.filters, axis=(1, -1, -2))
    .transpose((0, 2, 1)).reshape(B, O_C, _out_dim, _out_dim))
    + self.bias[:, np.newaxis, np.newaxis]

    return X

  def backward(self, loss):
    pass

  def call(self, x):
    pass
  

l1 = Conv2d(kernel_size=3, padding=0, stride=2, output_channel=6)
l1.forward(np.arange(3 * 4 ** 2).reshape(1,3,4,4))
