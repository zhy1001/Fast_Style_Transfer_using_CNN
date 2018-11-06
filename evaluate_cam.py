import os
import numpy as np
import scipy.misc
import tensorflow as tf
from argparse import ArgumentParser
import cv2

import helper
import residual

# 模型保存路径
CHECKPOINT_DIR = 'chinese-painting'
# 默认摄像头编号
VIDEO_DEVICE = 1
# 请勿修改以下部分


# 使输入图像经过训练好的网络
def ffwd_cam(cam_id, checkpoint_dir):
    cap = cv2.VideoCapture(cam_id)
    # print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 设定图片处理尺寸
    cam_shape = (240, 320, 3)
    # 读取拍摄的图片
    _, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = scipy.misc.imresize(frame_rgb, cam_shape)
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
        while True:
            _, frame = cap.read()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = scipy.misc.imresize(frame_rgb, cam_shape)
            X[0] = img
            _preds = sess.run(preds, feed_dict={img_placeholder: X})
            img_show = np.clip(_preds[0], 0, 255).astype(np.uint8)
            cv2.imshow("capture", cv2.cvtColor(img_show, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


def build_parser():
    parser = ArgumentParser()
    # 训练好的模型目录或文件名
    parser.add_argument('--checkpoint', type=str,
                        dest='checkpoint_dir',
                        help='checkpoints dir or .ckpt file',
                        metavar='CHECKPOINT', default=CHECKPOINT_DIR)

    parser.add_argument('--cam', type=int,
                        dest='cam', help='Camera ID',
                        metavar='VIDEO_DEVICE_NUMBER', default=VIDEO_DEVICE)

    return parser


def main():
    parser = build_parser()
    opts = parser.parse_args()

    ffwd_cam(opts.cam, opts.checkpoint_dir)


if __name__ == '__main__':
    main()
