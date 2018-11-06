import tensorflow as tf
import numpy as np

# 残差网络（参考论文）
def net(image):
    # 卷积：image, filterNumber, filterSize, Stride
    conv1 = _conv_layer(image, 32, 9, 1)
    conv2 = _conv_layer(conv1, 64, 3, 2)
    conv3 = _conv_layer(conv2, 128, 3, 2)
    # 残差模块
    resid1 = _residual(conv3, 3)
    resid2 = _residual(resid1, 3)
    resid3 = _residual(resid2, 3)
    resid4 = _residual(resid3, 3)
    resid5 = _residual(resid4, 3)
    # 反卷积
    conv_t1 = _conv_t_layer(resid5, 64, 3, 2)
    conv_t2 = _conv_t_layer(conv_t1, 64, 3, 2)
    # 卷积
    conv_t3 = _conv_layer(conv_t2, 3, 9, 1, relu=False)
    # 论文中提供
    preds = tf.nn.tanh(conv_t3)*150 + 255./2
    return preds


def _conv_layer(net, num_filters, filter_size, strides, relu=True):
    weights_init = _conv_init_vars(net, num_filters, filter_size)
    stride_shape = [1, strides, strides, 1]
    net = tf.nn.conv2d(net, weights_init, stride_shape, padding='SAME')
    # Batch Normalization
    net = _norm(net)
    if relu:
        net = tf.nn.relu(net)
    return net


def _conv_t_layer(net, num_filters, filter_size, strides):
    weights_init = _conv_init_vars(net, num_filters, filter_size, transpose=True)
    batch_size, rows, cols, in_channels = [i.value for i in net.get_shape()]
    new_rows, new_cols = int(rows*strides), int(cols*strides)
    new_shape = [batch_size, new_rows, new_cols, num_filters]
    # 转换成numpy数组
    tf_shape = tf.stack(new_shape)
    stride_shape = [1, strides, strides, 1]
    net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, stride_shape, padding='SAME')
    return tf.nn.relu(net)


def _residual(net, filter_size=3):
    tmp = _conv_layer(net, 128, filter_size, 1)
    return net + _conv_layer(tmp, 128, filter_size, 1, relu=False)


def _norm(net):
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]
    mu, sigma_sq = tf.nn.moments(net, [1, 2], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))
    epsilon = 1e-3
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)

    return scale * normalized + shift


def _conv_init_vars(net, out_channels, filter_size, transpose=False):
    _, rows, cols, in_channels = [i.value for i in net.get_shape()]
    if not transpose:
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
    else:
        weights_shape = [filter_size, filter_size, out_channels, in_channels]

    # 高斯初始化
    weights_init = tf.Variable(tf.truncated_normal(weights_shape, stddev=0.1, seed=1), dtype=tf.float32)
    return weights_init
