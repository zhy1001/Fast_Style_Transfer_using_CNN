import os
import numpy as np
import scipy.misc


# 用到的工具函数
# 读取图像
def read_img(path, img_size=False):
    img = scipy.misc.imread(path, mode='RGB')
    if not (len(img.shape) == 3 and img.shape[2] == 3):
        img = np.dstack((img, img, img))

    if img_size != False:
        img = scipy.misc.imresize(img, img_size)

    return img


# 返回某目录下的文件列表
def list_files(in_path):
    files = []
    # os.walk()返回（当前目录，当前目录下的文件夹，当前目录下的文件）
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        files.extend(filenames)
        # 若不break将继续遍历下级目录
        break
    return files


# 保存图像
def save_img(out_path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(out_path, img)
