import tensorflow as tf
import numpy as np
import scipy.io

# 使用VGGNet 19，若需使用其他版本，请按照网络结构自行修改
VGGNet_LAYERS = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
    'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
    'relu5_3', 'conv5_4', 'relu5_4'
)


# 从matlab文件加载训练好的VGGNet参数
def load_net(data_path):
    data = scipy.io.loadmat(data_path)
    # 图像均值存储位置
    mean = data['normalization'][0][0][0]
    # 在RGB各通道求均值
    mean_pixel = np.mean(mean, axis=(0, 1))
    # 读取权值
    weights = data['layers'][0]
    return weights, mean_pixel


# 从matlab文件加载训练好的VGGNet参数
def net(weights, input_image):
    net = {}
    current = input_image

    for i, name in enumerate(VGGNet_LAYERS):
        kind = name[:4]
        if kind == 'conv':
            # 读取各层各neuron权值和偏差
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            # kernels转换成tensorflow的数据格式
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            # bias从1行xx列转成一维数组
            bias = bias.reshape(-1)
            # 执行convolution
            current = _conv_layer(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current)
        elif kind == 'pool':
            current = _pool_layer(current)
        # net字典包含了图像经过VGGNet后各层的输出
        net[name] = current

    assert len(net) == len(VGGNet_LAYERS)
    return net


def _conv_layer(input, weights, bias):
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1), padding='SAME')
    return tf.nn.bias_add(conv, bias)


def _pool_layer(input):
    return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')


# VGGNet预处理，减去（训练得到的）均值
def preprocess(image, mean_pixel):
    return image - mean_pixel

