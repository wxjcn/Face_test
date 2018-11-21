from scipy import misc
import tensorflow as tf
import align.detect_face
import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image

import sys

print(sys.path)
def Dect_for_batch(path):#和单个检测重复   之后合并

    #对于根目录下的所有图片进行检测
    #检测结果保存在另一个_dect根目录  下的_dect子目录
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    gpu_memory_fraction = 1.0
    print('Creating networks and loading parameters')

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    for root, dirs, files in os.walk(path, topdown=False):
        #直接遍历所有文件
        for name in files:
            image_path = os.path.join(root, name)
            print(image_path)
            #检测
            try:
                img = misc.imread(image_path)
                bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
            except:
                pass
            #检测

            #保存

            #文件路径
            path_sub = image_path.split("\\")
            save_path_dir = ""
            for sub in path_sub[:-3]:
                save_path_dir += sub + "\\"
                pass
            save_path_dir += path_sub[-3] + "_dect\\"
            save_path_dir += path_sub[-2] + "_dect"
            image_name = path_sub[-1].split('.')

            if not os.path.exists(save_path_dir):  # 当文件夹不存在时创建
                os.makedirs(save_path_dir)
            # 文件路径

            count = 0
            for face_position in bounding_boxes:
                count += 1
                save_path = save_path_dir + "\\" + image_name[0] + str(count) + "." + image_name[-1]
                if not os.path.exists(save_path):
                    try:
                        face_position = face_position.astype(int)
                        cv2.rectangle(img, (face_position[0], face_position[1]), (face_position[2], face_position[3]),
                                      (0, 255, 0), 2)
                        crop = img[face_position[1]:face_position[3],
                               face_position[0]:face_position[2], ]
                        crop = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_CUBIC)
                        misc.imsave(save_path, crop)
                    except:
                        pass
                    pass
            pass
                # 保存
    pass

def Dect_For_One(path):
    #对于输入文件夹下的 图片创建对应的_dect文件夹保存检测后的照片
    #主要用于测试
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    gpu_memory_fraction = 1.0
    print('Creating networks and loading parameters')

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    img = misc.imread(path)
    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    faces_num = bounding_boxes.shape[0]
    crop_faces = []
    #拼接保存目录
    path_sub = path.split("\\")
    save_path_dir = ""
    for sub in path_sub[:-3]:
        save_path_dir += sub + "\\"
        pass
    save_path_dir += path_sub[-3] + "_dect\\"
    save_path_dir += path_sub[-2] + "_dect"

    if not os.path.exists(save_path_dir):#当文件夹不存在时创建
        os.makedirs(save_path_dir)
    count = 0
    try:
        for face_position in bounding_boxes:
            count += 1
            face_position = face_position.astype(int)
            cv2.rectangle(img, (face_position[0], face_position[1]), (face_position[2], face_position[3]), (0, 255, 0), 0)
            crop = img[face_position[1]:face_position[3],
                   face_position[0]:face_position[2], ]

            crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_CUBIC)
            image_name = path_sub[-1].split('.')
            save_path = save_path_dir + "\\" + image_name[0] + str(count) + "." + image_name[-1]
            misc.imsave(save_path, crop)
    except:
        pass
    plt.imshow(img)
    plt.show()
    cv2.imshow("tt", img)
    pass

def main():
    #写两个函数
    #一个批量处理、一个用作测试
    #Dect_For_One("D:\\Ada_Py\\t\\u=313189501,899734850&fm=26&gp=0.jpg")
    #注意用"\\"斜杠
    Dect_for_batch("./face/DB")
    pass

if __name__ == '__main__':
    main()
