
import numpy as np
from scipy import misc
import os.path

IMAGE_SIZE = 256

def load_to_tensorflow_for_dir(image_path, do_prewhiten=True):
    # 输入文件夹下
    image_paths = []
    for root, dirs, files in os.walk(image_path):
        for file in files:
            image_paths.append(os.path.join(root, file))
        pass
    nrof_samples = len(image_paths)
    images = np.zeros((nrof_samples, IMAGE_SIZE, IMAGE_SIZE, 3))
    for i in range(nrof_samples):
        img = misc.imread(image_paths[i])
        if img.ndim == 2:
            img = to_rgb(img)
        if do_prewhiten:
            img = prewhiten(img)
        images[i, :, :, :] = img
    return images
    pass

def load_to_tensorflow_for_one_from_path(image_path, do_prewhiten=True):
    #    输入单张图片
    #    图片尺寸96/128
    #
    images = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 3))
    img = misc.imread(image_path)
    if img.ndim == 2:
        img = to_rgb(img)
    if do_prewhiten:
        img = prewhiten(img)
    images[0, :, :, :] = img
    return images
    pass
"""""""""""
"""""""""""

def load_to_tensorflow_for_one(crop, do_prewhiten=True):
    #    输入单张图片
    #    图片尺寸96/128
    #    模型输入是四维的
    images = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 3))
    img = crop
    if img.ndim == 2:
        img = to_rgb(img)
    if do_prewhiten:
        img = prewhiten(img)
    images[0, :, :, :] = img
    return images
    pass

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y

def crop(image, random_crop, image_size):
    if image.shape[1] > image_size:
        sz1 = int(image.shape[1] // 2)
        sz2 = int(image_size // 2)
        if random_crop:
            diff = sz1 - sz2
            (h, v) = (np.random.randint(-diff, diff + 1), np.random.randint(-diff, diff + 1))
        else:
            (h, v) = (0, 0)
        image = image[(sz1 - sz2 + v):(sz1 + sz2 + v), (sz1 - sz2 + h):(sz1 + sz2 + h), :]
    return image

