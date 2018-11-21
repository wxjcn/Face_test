import tensorflow as tf
import Grape.data_process as data_process
import numpy as np
from scipy import misc
import os.path
from tensorflow.python.platform import gfile
import os
import time
import Grape.MessageShow as MessageShow
import datetime
import Grape.alig.align_dataset_mtcnn as alig
import cv2
from numba import jit
IMAGE_SIZE = alig.IMAGE_SIZE

#   DB_PATH = "D:\\Ada_Py\\tensorflow_test\\Face_test\\Grape\\face\\DB_dect"
DB_PATH = "./face/DB_dect"
vector_length = 2048
THRESGOLD = 19


def load_model():
    with tf.Graph().as_default() as graph:
        with tf.gfile.FastGFile('./R_model.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            images_placeholder, embeddings, phase_train_placeholder = tf.import_graph_def(
                graph_def=graph_def, name="",
                return_elements=[
                    "input:0",
                    "embeddings:0",
                    "phase_train:0"
                ]
            )
        pass
    return graph, images_placeholder, embeddings, phase_train_placeholder
    pass


def create_or_get_vector(sess, image_path, embeddings, images_placeholder, phase_train_placeholder):
    #   针对处理数据库数据
    #   将向量保存 以加快计算速度
    #   特征值以txt文档形式保存
    image_path_sub = image_path.split('dect')
    vector_dir = image_path_sub[0] + "vector" + image_path_sub[1] + "vector"
    vector_path = vector_dir + image_path_sub[2]
    vector_path = "." + (vector_path.split("."))[1] + ".txt"
    if not os.path.exists(vector_path):
        #
        # image_data = gfile.FastGFile(image_path, mode="rb").read()  # 二进制形式打开
        image_data = data_process.load_to_tensorflow_for_one_from_path(image_path)
        vector = sess.run(embeddings, feed_dict={images_placeholder: image_data, phase_train_placeholder: True})
        # 经过卷积神经网络处理的结果是一个多维数组，需要将这个结果压缩成一个一维数组
        # 数组常数为2048
        vector_values = np.squeeze(vector)
        vector_values = np.reshape(vector_values, newshape=[vector_length])
        #   将特征值保存
        vector_string = ','.join(str(sub_string) for sub_string in vector_values)
        if not os.path.exists(vector_dir):
            os.makedirs(vector_dir)
        with open(vector_path, 'a') as bottleneck_file:  # 打开文件并写入
            bottleneck_file.write(vector_string)
    else:
        with open(vector_path, 'r') as vector_file:
            vector_string = vector_file.read()
        vector_values = [float(sub_string) for sub_string in vector_string.split(',')]
        #   这里返回结果直接为2048维的列表
        #vector_values = np.reshape(vector_values, newshape=[1, 2048])
    return vector_values
    pass


def get_DB_vector(sess, images_placeholder, embeddings, phase_train_placeholder):
    #   把数据库里的数据全部调出，计算特征
    #   ？？？还是以此对比，对比成功就结束
    DB_vector = []
    DB_category_list = []
    append_v = DB_vector.append
    append_c = DB_category_list.append
    for root, dirs, _ in os.walk(DB_PATH):
        for dir in dirs:
            for _, _, files in os.walk(os.path.join(root, dir)):
                for file in files:
                    try:
                        image_path = os.path.join(os.path.join(root, dir), file)
                        image_vector = create_or_get_vector(sess, image_path, embeddings, images_placeholder,
                                                            phase_train_placeholder)
                        image_vector = np.array(image_vector)
                        """""""""""
                        temp = {
                            "vector": image_vector,
                            "category": dir.split("_dect")[0]
                        }
                        DB_vector.append(temp)
                        """""""""""
                        append_v(image_vector)
                        append_c(dir.split("_dect")[0])
                        pass
                    except:
                        pass
                pass
        pass
    return DB_vector, DB_category_list
    pass


@jit
def Compare(crop ,DB_vector, DB_category_list, sess, images_placeholder, embeddings, phase_train_placeholder):
    #   现在只输入一张图片
    #   处理新加进来的图片
    image_data = data_process.load_to_tensorflow_for_one(crop, True)
    time_for_vectorS = time.time()
    image_vector = sess.run(embeddings, feed_dict={images_placeholder: image_data, phase_train_placeholder: True})
    #   image_vector = np.reshape(image_vector, newshape=[2048])
    image_vector = np.reshape(image_vector, newshape=[vector_length])
    time_for_vectorE = time.time()
    tf.add_to_collection("time_for_vector", time_for_vectorE - time_for_vectorS)
    #
    #   Relu_dic = []
    #   改写成矩阵运算？？？
    """""""""
    for i in range(len(DB_vector)):
        Relu_dis = np.sum(
            np.square(
                np.subtract(image_vector, DB_vector[i]["vector"])
            )
        )
        #   现在为每张图片创建一个字典 与数据库中的图片全部比较
        #   之后考虑改为与每一个资料中的平均值？最小值？做比较？？？
        temp = {
            "category": DB_vector[i]["category"],
            "Relu_dis": Relu_dis
        }
        Relu_dic.append(temp)
        pass
    """""""""
    #   将本次的特征值广播为与加载数据库同维的矩阵
    #   image_vector_matrix = image_vector.repeat(len_of_DB, axis=1)

    time_for_searchS = time.time()
    Relu_dis_mat = np.sum(
        np.square(
            np.subtract(DB_vector, image_vector)
        ), axis=1
    )
    time_for_searchE = time.time()
    tf.add_to_collection("time_for_search", time_for_searchE - time_for_searchS)
    return Relu_dis_mat
    pass


def Any(Relu_mat, DB_category_list):
    #   不需要排序 只要找到最小的
    #   Relu_dic = sorted(Relu_dic, key=lambda dis: dis["Relu_dis"])
    #   min_dic = min(Relu_dic, key=lambda dis: dis["Relu_dis"])
    min_dic = np.min(Relu_mat)
    if min_dic < THRESGOLD:
        Relu_mat = list(Relu_mat)
        index = Relu_mat.index(min_dic)
        return DB_category_list[index]
    else:
        return "No Match"
    pass


def Compare_for_more(crops, sizes ,DB_vector, DB_category_list, sess, images_placeholder, embeddings, phase_train_placeholder):
    Match_Result = []
    """""""""""
        for root, _, files in os.walk(dir_path):
        for file in files:
            path = os.path.join(root, file)
            Relu_dic = Compare(path, DB_vector, sess, images_placeholder, embeddings, phase_train_placeholder)
            Result = Any(Relu_dic)
            if not Result == "No Match":
                #   [检测名称， 尺寸]
                Match_Result.append([Result, file.split('.')[0]])
            pass
    """""""""""
    for i in range(len(crops)):
        #   返回结果为表示距离的列表，和DB_category_list在下标上对应
        #   img = img[..., ::-1]
        Relu_mat = Compare(crops[i], DB_vector, DB_category_list, sess, images_placeholder, embeddings, phase_train_placeholder)
        Result = Any(Relu_mat, DB_category_list)
        if not Result == "No Match":
            #   [检测名称， 尺寸]
            Match_Result.append([Result, sizes[i]])
        pass
    return Match_Result
    pass


def remove_dir(path_dir):
    for root, _, files in os.walk(path_dir):
        for file in files:
            os.remove(os.path.join(root, file))
    pass


def Any_for_Match(Match_list, Match_Result, Messagehow_dic):
    #   Match_list  总的识别结果的保存
    #   Match_list  单次的识别结果
    #   包括消息的传送也在这里实现
    for Result in Match_Result:
        if not Result[0] in Match_list.keys():
            Match_list[Result[0]] = []
            Match_list[Result[0]].append(int(Result[1]))
            pass
        else:
            #   将本次检测到的尺寸加进去
            Match_list[Result[0]].append(int(Result[1]))
            #   判断行进方向
            length = len(Match_list[Result[0]])
            if length > 2:
                #   用至少三张照片判定
                #   用户有向前行进的倾向
                step_sum = 0
                for i in range(2):
                    step_sum += Match_list[Result[0]][length - 1 - i] - Match_list[Result[0]][length - 2 - i]
                    pass
                if step_sum > 800:
                    #   用户有向前行进的倾向
                    #   实际调用函数显示用户信息
                    #   比如在某一时间段内不重复显示同一个用户的信息
                    #   print(Result[0])
                    time = datetime.datetime.now()
                    if Result[0] in Messagehow_dic.keys():
                        #   非首次触发
                        #   判断时间，保持在一分钟内不重复显示
                        lasttime = Messagehow_dic[Result[0]]
                        #   更新时间
                        Messagehow_dic[Result[0]] = time
                        time_difference_sub = str(time - lasttime).split(':')
                        time_difference = float(time_difference_sub[-3]) * 60 + float(time_difference_sub[-2]) + float(time_difference_sub[-1])/60
                        #
                        #   时间算错了
                        #
                        if time_difference > 3:
                            MessageShow.ShowInfo(Result[0])
                        pass
                    else:
                        #   首次触发
                        #   加入时间并调用函数显示
                        Messagehow_dic[Result[0]] = time
                        MessageShow.ShowInfo(Result[0])
                        pass
                    pass
                pass
            pass
        pass
    return Match_list
    pass


def main():
    graph, images_placeholder, embeddings, phase_train_placeholder = load_model()
    #   从视频流获取图片然后调用
    with tf.Session(graph=graph) as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        DB_vector = get_DB_vector(sess, images_placeholder, embeddings, phase_train_placeholder)
        pass
    pass


if __name__ == '__main__':
    main()