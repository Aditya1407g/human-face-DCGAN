import math
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

class batch_norm(object):
    """
    Batch normalization layer.
    Code modification of http://stackoverflow.com/a/33950177
    """
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
            self.name = name

    def __call__(self, x, train=True):
        shape = x.get_shape().as_list()
        if train:
            with tf.variable_scope(self.name) as scope:
                self.beta = tf.get_variable("beta", [shape[-1]],
                                            initializer=tf.constant_initializer(0.0))
                self.gamma = tf.get_variable("gamma", [shape[-1]],
                                             initializer=tf.random_normal_initializer(1.0, 0.02))
                batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
                # Apply moving average to mean and variance
                ema_apply_op = self.ema.apply([batch_mean, batch_var])
                self.ema_mean, self.ema_var = self.ema.average(batch_mean), self.ema.average(batch_var)
                with tf.control_dependencies([ema_apply_op]):
                    mean, var = tf.identity(batch_mean), tf.identity(batch_var)
        else:
            # Use stored moving averages
            mean = tf.get_variable("moving_mean", [shape[-1]],
                                   initializer=tf.constant_initializer(0.0),
                                   trainable=False)
            var = tf.get_variable("moving_variance", [shape[-1]],
                                  initializer=tf.constant_initializer(1.0),
                                  trainable=False)
        normed = tf.nn.batch_normalization(x, mean, var, self.beta, self.gamma, self.epsilon)
        normed.set_shape(x.get_shape())
        return normed

def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1):
    """
    2D convolution with support for groups.
    Adapted from: https://github.com/ethereon/caffe-tensorflow
    """
    input_channels = int(x.get_shape()[-1])
    # Lambda to perform a convolution operation
    convolve = lambda i, k: tf.nn.conv2d(i, k, strides=[1, stride_y, stride_x, 1], padding=padding)
    with tf.variable_scope(name) as scope:
        # Define weight and bias variables
        weights = tf.get_variable('weights', shape=[filter_height, filter_width, input_channels // groups, num_filters],
                                  initializer=tf.truncated_normal_initializer(stddev=0.02))
        biases = tf.get_variable('biases', shape=[num_filters],
                                 initializer=tf.constant_initializer(0.0))
        if groups == 1:
            conv_out = convolve(x, weights)
        else:
            # Split input and weights for group convolution
            input_groups = tf.split(x, num_or_size_splits=groups, axis=3)
            weight_groups = tf.split(weights, num_or_size_splits=groups, axis=3)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]
            conv_out = tf.concat(output_groups, axis=3)
        conv_out = tf.nn.bias_add(conv_out, biases)
        return tf.nn.relu(conv_out, name=scope.name)

def fc(x, num_in, num_out, name, relu=True):
    """
    Fully connected layer.
    """
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[num_in, num_out],
                                  initializer=tf.truncated_normal_initializer(stddev=0.02))
        biases = tf.get_variable('biases', shape=[num_out],
                                 initializer=tf.constant_initializer(0.0))
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)
        return tf.nn.relu(act) if relu else act

def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
    """
    2D max pooling.
    """
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)

def lrn(x, radius, alpha, beta, name, bias=1.0):
    """
    Local response normalization.
    """
    return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha,
                                              beta=beta, bias=bias, name=name)

def dropout(x, keep_prob):
    """
    Dropout layer.
    """
    return tf.nn.dropout(x, keep_prob)

def conv1d(input, filter_width, out_channels, in_channels=None, stride=1,
           HeUniform=False, with_bias=True, name=None):
    """
    1D convolution layer.
    """
    with tf.variable_scope(name):
        if in_channels is None:
            in_channels = input.get_shape()[-1]
        std = np.sqrt(2.0 / (filter_width * out_channels)) if HeUniform else 0.02
        kernel = tf.get_variable('weights', [filter_width, in_channels, out_channels],
                                 initializer=tf.random_normal_initializer(stddev=std))
        conv = tf.nn.conv1d(input, kernel, stride=stride, padding='SAME')
        if with_bias:
            biases = tf.get_variable('b', [out_channels], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)
        return conv

def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           padding='SAME', name="conv2d"):
    """
    2D convolution layer.
    """
    with tf.variable_scope(name):
        w = tf.get_variable('weights', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)
        biases = tf.get_variable('biases', [output_dim],
                                 initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)
        return conv

def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d"):
    """
    2D deconvolution (transposed convolution) layer.
    """
    with tf.variable_scope(name):
        w = tf.get_variable('weights', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                        strides=[1, d_h, d_w, 1])
        biases = tf.get_variable('biases', [output_shape[-1]],
                                 initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.bias_add(deconv, biases)
        return deconv

def lrelu(x, leak=0.2, name="lrelu"):
    """
    Leaky ReLU activation function.
    """
    return tf.maximum(x, leak * x, name=name)

def linear(input_, output_size, name, stddev=0.02, bias_start=0.0):
    """
    Linear (fully connected) layer.
    """
    shape = input_.get_shape().as_list()
    input_ = tf.reshape(input_, [shape[0], -1])
    with tf.variable_scope(name):
        matrix = tf.get_variable("weights", [input_.get_shape()[1], output_size],
                                 initializer=tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        return tf.matmul(input_, matrix) + bias

def unravel_argmax(argmax, shape):
    """
    Helper to convert flattened indices into 2D indices.
    """
    output_list = [argmax // (shape[2] * shape[3]),
                   (argmax % (shape[2] * shape[3])) // shape[3]]
    return tf.stack(output_list)

def unpool_layer2x2_batch(bottom, argmax):
    """
    Unpooling layer using sparse tensor operations.
    """
    bottom_shape = tf.shape(bottom)
    top_shape = [bottom_shape[0], bottom_shape[1]*2, bottom_shape[2]*2, bottom_shape[3]]
    batch_size, height, width, channels = top_shape
    argmax_shape = tf.to_int64([batch_size, height, width, channels])
    argmax = unravel_argmax(argmax, argmax_shape)
    t1 = tf.tile(tf.reshape(tf.range(channels), [1, channels]), [batch_size * (width//2) * (height//2), 1])
    t1 = tf.reshape(t1, [channels, batch_size, height//2, width//2, 1])
    t1 = tf.transpose(t1, [1, 0, 2, 3, 4])
    t2 = tf.tile(tf.reshape(tf.range(batch_size), [1, batch_size]), [channels * (width//2) * (height//2), 1])
    t2 = tf.reshape(t2, [batch_size, channels, height//2, width//2, 1])
    t3 = tf.transpose(argmax, [1, 4, 2, 3, 0])
    t = tf.concat(4, [t2, t3, t1])
    indices = tf.reshape(t, [batch_size * channels * (height//2) * (width//2), 4])
    x1 = tf.transpose(bottom, [0, 3, 1, 2])
    values = tf.reshape(x1, [-1])
    delta = tf.SparseTensor(indices, values, tf.to_int64(top_shape))
    return tf.sparse_tensor_to_dense(tf.sparse_reorder(delta))

def UnPooling2x2ZeroFilled(x):
    """
    Simple 2x2 unpooling by concatenating zeros.
    """
    out = tf.concat(3, [x, tf.zeros_like(x)])
    out = tf.concat(2, [out, tf.zeros_like(out)])
    sh = x.get_shape().as_list()
    if None not in sh[1:]:
        return tf.reshape(out, [-1, sh[1]*2, sh[2]*2, sh[3]])
    else:
        shv = tf.shape(x)
        ret = tf.reshape(out, tf.stack([-1, shv[1]*2, shv[2]*2, sh[3]]))
        ret.set_shape([None, None, None, sh[3]])
        return ret

def FixedUnPooling(x, shape, unpool_mat=None):
    """
    Unpool the input with a fixed matrix using Kronecker product.
    """
    if shape[0] == 2 and shape[1] == 2 and unpool_mat is None:
        return UnPooling2x2ZeroFilled(x)
    input_shape = x.get_shape().as_list()
    if unpool_mat is None:
        mat = np.zeros(shape, dtype='float32')
        mat[0][0] = 1
        unpool_mat = tf.constant(mat, name='unpool_mat')
    elif isinstance(unpool_mat, np.ndarray):
        unpool_mat = tf.constant(unpool_mat, name='unpool_mat')
    assert unpool_mat.get_shape().as_list() == list(shape)
    fx = flatten(tf.transpose(x, [0, 3, 1, 2]))
    fx = tf.expand_dims(fx, -1)
    mat = tf.expand_dims(flatten(unpool_mat), 0)
    prod = tf.matmul(fx, mat)
    prod = tf.reshape(prod, [-1, input_shape[3], input_shape[1], input_shape[2], shape[0], shape[1]])
    prod = tf.transpose(prod, [0, 2, 4, 3, 5, 1])
    prod = tf.reshape(prod, [-1, input_shape[1]*shape[0], input_shape[2]*shape[1], input_shape[3]])
    return prod

def flatten(x):
    """
    Flatten the input tensor.
    """
    return tf.reshape(x, [-1])

def binary_cross_entropy(preds, targets, name=None):
    """
    Computes binary cross entropy loss.
    """
    eps = 1e-12
    with ops.op_scope([preds, targets], name, "bce_loss") as name:
        preds = ops.convert_to_tensor(preds, name="preds")
        targets = ops.convert_to_tensor(targets, name="targets")
        return tf.reduce_mean(-(targets * tf.log(preds + eps) + (1. - targets) * tf.log(1. - preds + eps)))

def conv_cond_concat(x, y):
    """
    Concatenate conditioning vector on feature map axis.
    """
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat(3, [x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])])

def mse(pred, target):
    """
    Compute mean squared error loss.
    """
    num = pred.get_shape().as_list()[0]
    pred = tf.reshape(pred, [num, -1])
    target = tf.reshape(target, [num, -1])
    mse_sum = tf.reduce_sum(tf.pow(pred - target, 2.0), 1)
    mse_loss = tf.reduce_mean(mse_sum)
    return mse_loss

def instance_norm(x, name):
    """
    Instance normalization layer.
    """
    with tf.variable_scope(name):
        epsilon = 1e-5
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        scale = tf.get_variable('scale', [x.get_shape()[-1]],
                                initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
        offset = tf.get_variable('offset', [x.get_shape()[-1]], initializer=tf.constant_initializer(0.0))
        out = scale * tf.div(x - mean, tf.sqrt(var + epsilon)) + offset
    return out

def l1_loss(pred, target):
    """
    Compute L1 loss.
    """
    return tf.reduce_mean(tf.abs(pred - target))

def l2_loss(pred, target):
    """
    Compute L2 loss.
    """
    return 0.5 * tf.reduce_mean(tf.pow(pred - target, 2.0))

def tv_loss(images):
    """
    Total variation loss.
    """
    return tf.reduce_mean(tf.image.total_variation(images))

def mmd_loss(source, target):
    """
    Maximum mean discrepancy loss between source and target.
    """
    source_mean = tf.reduce_mean(source)
    target_mean = tf.reduce_mean(target)
    return 0.5 * tf.reduce_sum(tf.pow(source_mean - target_mean, 2.0))
