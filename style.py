import os
from argparse import ArgumentParser

import train
import helper

# 从论文里搞出来的一些参数
CONTENT_WEIGHT = 1e1
STYLE_WEIGHT = 1e2
# 提高图像真实性（参考论文）
TV_WEIGHT = 2e2

LEARNING_RATE = 5e-3
NUM_EPOCHS = 2
# 模型保存路径
CHECKPOINT_DIR = 'checkpoints'
# 每隔多少次存一次model
CHECKPOINT_ITERATIONS = 200
# 训练好的VGGNet的参数
VGG_PATH = 'imagenet-vgg-verydeep-19.mat'
# 用于训练的图片的所在位置
TRAIN_PATH = 'training_images'
BATCH_SIZE = 5
# 请勿修改以下部分


# 导入运行参数
def build_parser():
    parser = ArgumentParser()

    parser.add_argument('--train-path', type=str,
            dest='train_path', help='training images',
            metavar='TRAIN_PATH', default=TRAIN_PATH)
    parser.add_argument('--style', type=str,
            dest='style', help='style image',
            metavar='STYLE', required=True)

    parser.add_argument('--checkpoint-dir', type=str,
            dest='checkpoint_dir', help='checkpoint path',
            metavar='CHECKPOINT_DIR', default=CHECKPOINT_DIR)

    parser.add_argument('--vgg-path', type=str,
                        dest='vgg_path', help='path to trained VGG parameters (default %(default)s)',
                        metavar='VGG_PATH', default=VGG_PATH)
    parser.add_argument('--content-weight', type=float,
                        dest='content_weight', help='content weight (default %(default)s)',
                        metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)
    parser.add_argument('--style-weight', type=float,
                        dest='style_weight', help='style weight (default %(default)s)',
                        metavar='STYLE_WEIGHT', default=STYLE_WEIGHT)
    parser.add_argument('--tv-weight', type=float,
                        dest='tv_weight', help='total variation regularization weight (default %(default)s)',
                        metavar='TV_WEIGHT', default=TV_WEIGHT)

    parser.add_argument('--epochs', type=int,
                        dest='epochs', help='number of epochs',
                        metavar='EPOCHS', default=NUM_EPOCHS)
    parser.add_argument('--batch-size', type=int,
                        dest='batch_size', help='batch size',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)
    parser.add_argument('--learning-rate', type=float,
                        dest='learning_rate',
                        help='learning rate (default %(default)s)',
                        metavar='LEARNING_RATE', default=LEARNING_RATE)

    return parser


def _get_files(img_dir):
    files = helper.list_files(img_dir)
    return [os.path.join(img_dir, x) for x in files]


def main():
    parser = build_parser()
    options = parser.parse_args()

    # 读取图片
    style_target = helper.read_img(options.style)
    content_targets = _get_files(options.train_path)

    args = [
        content_targets,
        style_target,
        options.content_weight,
        options.style_weight,
        options.tv_weight,
        options.vgg_path
    ]

    kwargs = {
        "epochs": options.epochs,
        "print_iterations": CHECKPOINT_ITERATIONS,
        "batch_size": options.batch_size,
        "learning_rate": options.learning_rate,
        "save_path": os.path.join(options.checkpoint_dir, 'cnn-style.ckpt')
    }

    for preds, losses, i, epoch, delta_time in train.train(*args, **kwargs):
        style_loss, content_loss, tv_loss, loss = losses

        print('Epoch %d, Iteration: %d, Loss: %s Time: %.2fs' % (epoch, i, loss, delta_time))
        print('Loss: style: %s, content:%s, tv: %s' % (style_loss, content_loss, tv_loss))

    # 显示训练好的模型
    ckpt_dir = options.checkpoint_dir
    cmd_text = 'python3 evaluate.py --checkpoint %s ...' % ckpt_dir
    print("Training complete. For evaluation:\n    `%s`" % cmd_text)


if __name__ == '__main__':
    main()
