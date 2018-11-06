import os
import numpy as np
import tensorflow as tf
from argparse import ArgumentParser

import helper
import residual

# 模型保存路径
CHECKPOINT_DIR = 'checkpoints'
# 请勿修改以下部分


# 使输入图像经过训练好的网络
def ffwd(data_in, path_out, checkpoint_dir):
    img = helper.read_img(data_in)
    # 仅一张图片
    batch_size = 1

    with tf.Graph().as_default(), tf.Session() as sess:
        batch_shape = (batch_size,) + img.shape
        img_placeholder = tf.placeholder(tf.float32, shape=batch_shape, name='img_placeholder')
        preds = residual.net(img_placeholder)
        saver = tf.train.Saver()
        # 判断提供的checkpoints是目录还是文件
        if os.path.isdir(checkpoint_dir):
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise Exception("No checkpoint found...")
        else:
            saver.restore(sess, checkpoint_dir)

        # 开始处理图片
        X = np.zeros(batch_shape, dtype=np.float32)
        X[0] = img
        _preds = sess.run(preds, feed_dict={img_placeholder: X})
        helper.save_img(path_out, _preds[0])


def build_parser():
    parser = ArgumentParser()
    # 训练好的模型目录或文件名
    parser.add_argument('--checkpoint', type=str,
                        dest='checkpoint_dir',
                        help='checkpoints dir or .ckpt file',
                        metavar='CHECKPOINT', default=CHECKPOINT_DIR)

    parser.add_argument('--image', type=str,
                        dest='image', help='image to transform',
                        metavar='IMAGE_NAME', required=True)
    parser.add_argument('--output', type=str,
                        dest='output', help='image path (or file) to save',
                        metavar='OUTPUT_NAME', required=True)

    return parser


def main():
    parser = build_parser()
    opts = parser.parse_args()

    # 输出参数为目录的时候可自动识别
    if os.path.exists(opts.output) and os.path.isdir(opts.output):
        out_path = os.path.join(opts.output, os.path.basename(opts.image))
    else:
        out_path = opts.output

    ffwd(opts.image, out_path, opts.checkpoint_dir)


if __name__ == '__main__':
    main()
