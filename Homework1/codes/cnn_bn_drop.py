#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 17:22:31 2019

@author: chenchenma
"""

import numpy as np

from layers import *


class ConvNet_BND(object):
  """
  A convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - fc - softmax

  You may also consider adding dropout layer or batch normalization layer. 
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(1, 28, 28), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim
    H_prime = H - filter_size + 1 
    W_prime = W - filter_size + 1
    len_fc = int(num_filters * H_prime * W_prime / 4)
    
    
    W1 = np.random.normal(scale = weight_scale, size = (num_filters, C, filter_size, filter_size))
    W2 = np.random.normal(scale = weight_scale, size = (len_fc, hidden_dim))
    W3 = np.random.normal(scale = weight_scale, size = (hidden_dim, num_classes)) 
    
    b1 = 0
    b2 = 0
    b3 = 0
    
    bn_param = {'mode' : 'train', 'eps' : 1e-5, 'momentum' : 0.9}
    dropout_param = {'p' : 0.8, 'mode' : 'train'}
    
    self.bn_param = bn_param
    self.dropout_param = dropout_param
    
    self.params['W1'] = W1
    self.params['b1'] = b1
    self.params['W2'] = W2
    self.params['b2'] = b2
    self.params['W3'] = W3
    self.params['b3'] = b3
    self.params['gamma'] = np.ones(num_filters * H_prime * W_prime)
    self.params['beta'] = np.zeros(num_filters * H_prime * W_prime)

    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    #for k, v in self.params.iteritems():
      #self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    gamma, beta = self.params['gamma'], self.params['beta']
    
    #batchnorm param
    bn_param = self.bn_param
    
    #dropout param
    dropout_param = self.dropout_param
   
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    
    if y is None:
        
        bn_param['mode'] = 'test'
        dropout_param['mode'] = 'test'
        
    else:
        bn_param['mode'] = 'train'
        dropout_param['mode'] = 'train'
    
    out_conv, cache_conv = conv_forward(X, W1)
    out_conv_flat = out_conv.reshape(X.shape[0],-1)
    out_bn, cache_bn = batchnorm_forward(out_conv_flat, gamma, beta, bn_param)
    out_bn_unflat = out_bn.reshape(out_conv.shape)
    out_relu, cache_relu = relu_forward(out_bn_unflat)
    #out_relu_flat = out_relu.reshape(X.shape[0],-1)
    out_max, cache_max = max_pool_forward(out_relu, pool_param)
    out_max_flat = out_max.reshape(X.shape[0],-1)
    out_fc1, cache_fc1 = dropout_forward(out_max_flat, dropout_param)
    out_drop, cache_drop = fc_forward(out_fc1, W2, b2)
    out_relu2, cache_relu2 = relu_forward(out_drop)    
    out_fc2, cache_fc2 = fc_forward(out_relu2, W3, b3)
    scores = out_fc2
    
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################

    loss, dout_fc2 = softmax_loss(scores, y)
    dout_relu2, dW3, db3 = fc_backward(dout_fc2, cache_fc2) 
    dout_drop = relu_backward(dout_relu2, cache_relu2)
    dout_fc1, dW2, db2 = fc_backward(dout_drop, cache_drop) 
    dout_max_flat = dropout_backward(dout_fc1,cache_fc1)  
    dout_max = dout_max_flat.reshape(out_max.shape)
    dout_relu_flat = max_pool_backward(dout_max, cache_max)
    dout_relu = dout_relu_flat.reshape(out_relu.shape)
    dout_bn = relu_backward(dout_relu, cache_relu)
    dout_relu_flat = dout_bn.reshape(out_bn.shape) 
    dout_conv_unflat, dgamma, dbeta = batchnorm_backward(dout_relu_flat, cache_bn)
    dout_conv = dout_conv_unflat.reshape(out_conv.shape)   
    dx, dW1 = conv_backward(dout_conv, cache_conv)
    
    reg = self.reg
    dW1 = dW1 + reg * W1
    dW2 = dW2 + reg * W2
    dW3 = dW3 + reg * W3
    
    grads['W1'] = dW1
    grads['W2'] = dW2
    grads['W3'] = dW3
    
    grads['b1'] = np.zeros_like(b1)
    grads['b2'] = db2
    grads['b3'] = db3
    
    grads['gamma'] = dgamma
    grads['beta'] = dbeta
    
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  

