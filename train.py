import time
import functools

import tensorflow as tf
import numpy as np

import vgg
import residual
import helper

# 从论文里搞出来的一些参数
STYLE_LAYER = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER = 'relu4_2'


def train(content_targets, style_target, content_weight, style_weight, tv_weight,
          vgg_path, epochs=2, print_iterations=1000,
          batch_size=4, learning_rate=1e-3,
          save_path='model/style.ckpt'):
    # 根据batch丢弃最后的训练图像
    mod = len(content_targets) % batch_size
    if mod > 0:
        content_targets = content_targets[:-mod]

    style_features = {}
    # 训练图像大小：320x320x3，按照tensorflow格式
    batch_shape = (batch_size, 320, 320, 3)
    style_shape = (1,) + style_target.shape

    # 读取训练好的VGGNet模型
    weights, mean_pixel = vgg.load_net(vgg_path)

    with tf.Graph().as_default(), tf.Session() as sess:
        style_image = tf.placeholder(tf.float32, shape=style_shape, name='style_image')
        # 没看错！空图片减去均值
        style_image_pre = vgg.preprocess(style_image, mean_pixel)
        net = vgg.net(weights, style_image_pre)
        # 把style图片展开形成数组
        style_pre = np.array([style_target])
        for layer in STYLE_LAYER:
            # 取出该层的计算结果
            features = net[layer].eval(feed_dict={style_image: style_pre})
            # 行数为该层的Filter数（参见论文）
            features = np.reshape(features, (-1, features.shape[3]))
            # Gram Matrix: A'A (参见论文)
            gram = np.matmul(features.T, features) / features.size
            style_features[layer] = gram

    with tf.Graph().as_default(), tf.Session() as sess:
        x_content = tf.placeholder(tf.float32, shape=batch_shape, name='x_content')
        x_pre = vgg.preprocess(x_content, mean_pixel)

        content_features = {}
        content_net = vgg.net(weights, x_pre)
        # 同上，提取所需层
        content_features[CONTENT_LAYER] = content_net[CONTENT_LAYER]

        # 使用残差网络
        preds = residual.net(x_content/255.0)
        preds_pre = vgg.preprocess(preds, mean_pixel)
        net = vgg.net(weights, preds_pre)
        # 计算每个batch里的所有数据
        content_size = _tensor_size(content_features[CONTENT_LAYER])*batch_size
        assert _tensor_size(content_features[CONTENT_LAYER]) == _tensor_size(net[CONTENT_LAYER])
        # 计算经过残差网络和不经过时的差别
        content_loss = content_weight * (2*tf.nn.l2_loss(
            net[CONTENT_LAYER]-content_features[CONTENT_LAYER])/content_size
                                         )

        # 计算经过残差网络的图像与style图像之间的差别
        style_losses = []
        for style_layer in STYLE_LAYER:
            layer = net[style_layer]
            bs, height, width, filters = map(lambda i: i.value, layer.get_shape())
            size = height * width * filters
            feats = tf.reshape(layer, (bs, height * width, filters))
            feats_T = tf.transpose(feats, perm=[0, 2, 1])
            # Gram Matrix: A'A (参见论文)
            grams = tf.matmul(feats_T, feats) / size
            style_gram = style_features[style_layer]
            style_losses.append(2 * tf.nn.l2_loss(grams - style_gram)/style_gram.size)
        style_loss = style_weight * functools.reduce(tf.add, style_losses) / batch_size

        # 去图像噪声: Total Variation
        tv_y_size = _tensor_size(preds[:, 1:, :, :])
        tv_x_size = _tensor_size(preds[:, :, 1:, :])
        y_tv = tf.nn.l2_loss(preds[:, 1:, :, :] - preds[:, :batch_shape[1] - 1, :, :])
        x_tv = tf.nn.l2_loss(preds[:, :, 1:, :] - preds[:, :, :batch_shape[2] - 1, :])
        tv_loss = tv_weight * 2 * (x_tv / tv_x_size + y_tv / tv_y_size) / batch_size

        # 最终的loss函数
        loss = content_loss + style_loss + tv_loss

        # 开始训练过程
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            num_examples = len(content_targets)
            iterations = 0
            start_time = time.time()
            # 每一次epoch就用训练集的所有图片训练一遍
            while iterations * batch_size < num_examples:
                curr = iterations * batch_size
                step = curr + batch_size
                X_batch = np.zeros(batch_shape, dtype=np.float32)
                for j, img_p in enumerate(content_targets[curr:step]):
                   X_batch[j] = helper.read_img(img_p, (320, 320, 3)).astype(np.float32)

                iterations += 1
                # 确保每批次计算的时候不出错
                assert X_batch.shape[0] == batch_size
                feed_dict = {x_content: X_batch}

                # 开始训练
                train_step.run(feed_dict=feed_dict)

                # 隔几次打印一次训练进度
                is_print_iter = int(iterations) % print_iterations == 0
                # 是否是最后一个epoch
                is_last = epoch == epochs - 1 and iterations * batch_size >= num_examples

                # 打印信息
                should_print = is_print_iter or is_last
                if should_print:
                    current_time = time.time()
                    delta_time = current_time - start_time
                    start_time = current_time

                    to_get = [style_loss, content_loss, tv_loss, loss, preds]
                    test_feed_dict = {x_content: X_batch}

                    tup = sess.run(to_get, feed_dict=test_feed_dict)
                    _style_loss, _content_loss, _tv_loss, _loss, _preds = tup

                    losses = (_style_loss, _content_loss, _tv_loss, _loss)

                    saver = tf.train.Saver()
                    res = saver.save(sess, save_path)
                    yield (_preds, losses, iterations, epoch, delta_time)


def _tensor_size(tensor):
    from operator import mul
    return functools.reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)
