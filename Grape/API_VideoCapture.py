
import Grape.inference as inference
import Grape.alig.align_dataset_mtcnn as align

import cv2
import tensorflow as tf
import time

TEST_NUM = 2293063624
BATCH_TEST_NUM = 50

def Cam(DB_vector, DB_category_list, sess, images_placeholder, embeddings, phase_train_placeholder, pnet, rnet, onet ):
    #   启动摄像头
    print("*********************    Successfully open the camera  *********************")
    #   inference.THRESGOLD = float(input("Input the THRESGOLD："))
    #   interval = 0.001    #   睡眠时间
    Match_list = {}    #   将检测出的用户暂存在列表里
    Messagehow_dic = {}    #   确认输出的用户，暂存的字典，包括检测时间
    index = 0
    #   cam = Device(devnum=0)
    cam = cv2.VideoCapture(0)
    #   cam.set(3, 1280)
    #   cam.set(4, 720)
    #   cam.setResolution(width=680, height=420)
    timeC_start = time.time()
    while True:
        try:
            #
            #   cam.saveSnapshot('./temp_image/Cam_image/temp_image.jpg')
            ret, img = cam.read()
            #   cv2.imshow("capture", img)
            #   img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_CUBIC)
            #   对图片进行检测和裁剪
            #   crops 对应裁剪后的图片
            #   sizes 对应图片尺寸，一一对应

            time_for_dect = time.time()
            crops, sizes, img = align.Dect_For_One(img, pnet, rnet, onet)
            time_for_dectS = time.time()

            time_for_show = time.time()
            img = cv2.resize(img, (960, 720), interpolation=cv2.INTER_CUBIC)
            cv2.imshow("Capture", img)
            cv2.waitKey(1)
            time_for_showS = time.time()

            tf.add_to_collection("time_for_dect", time_for_dectS - time_for_dect)
            tf.add_to_collection("time_for_show", time_for_showS - time_for_show)

            #
            if len(crops) > 0:
            #if align.Dect_For_One('./temp_image/Cam_image/temp_image.jpg', pnet, rnet, onet) > 0:
                #   当检测出结果
                #   对所有裁剪结果匹配
                #dect_dir = "./temp_image/dect_image"
                #
                Match_Result = inference.Compare_for_more(crops, sizes, DB_vector, DB_category_list, sess, images_placeholder, embeddings,
                                                          phase_train_placeholder)
                #   返回结果包括[名字，尺寸]
                if len(Match_Result) > 0:
                    Match_list = inference.Any_for_Match(Match_list, Match_Result, Messagehow_dic)
                #   删除临时图片
                #   os.remove('./temp_image/Cam_image/temp_image.jpg')
                #   inference.remove_dir("./temp_image/dect_image")

            #   睡眠等待
            #   time_remaining = interval-time.time()%interval
            #   time.sleep(time_remaining)

            """""""""
            #   清零检测结果
            index += 1
            if index % 2000 == 0:
                Match_list.clear()
                index = 0
                pass
            """""""""
            index += 1
            if index == TEST_NUM:
                break
            if index % BATCH_TEST_NUM == 0:
                #   删除部分Match_list以保证在识别的帧数在相近时间内
                #   有待商榷
                #   每次删除四分之三的结果
                #print(len(Match_list["wxj"]))
                for key in Match_list.keys():
                    if len(Match_list[key]) > 1:
                        num_for_del = int(len(Match_list[key]) * 3 / 4)
                        for i in range(num_for_del):
                            del Match_list[key][0]
                        pass
                #   部分删除
                timeC_end = time.time()
                print('\n\nPer Cost:', round((timeC_end - timeC_start) / index, 4))
                #   搜索时间
                time_for_search = sum(tf.get_collection("time_for_search"))
                time_for_show = sum(tf.get_collection("time_for_show"))
                time_for_dect = sum(tf.get_collection("time_for_dect"))
                time_for_vector = sum(tf.get_collection("time_for_vector"))

                print('Per Cost for  show:', round((time_for_show) / index, 4))
                print('Per Cost for  dect:', round((time_for_dect) / index, 4))
                print('Per Cost for  vector:', round((time_for_vector) / index, 4))
                print('Per Cost for  search:', round((time_for_search) / index, 4))
                print("The other Cost:", round((
                                                           timeC_end - timeC_start - time_for_vector - time_for_dect - time_for_show - time_for_search) / index,
                                               4))
                #   MTCNN时间
                #   Res—Net时间
                pass
        except:
            print("ERROR")
            pass
    pass


def main():
    #   先启动模块
    #   加载神经网络
    graph, images_placeholder, embeddings, phase_train_placeholder = inference.load_model()
    pnet, rnet, onet = align.load_ali_net()
    print("*********************    Neural network was successfully loaded  *********************")
    #   启动会话
    with tf.Session(graph=graph) as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        #   启动获取数据库的全体特征向量
        #   将DB_vector作为两部分返回
        DB_vector, DB_category_list = inference.get_DB_vector(sess, images_placeholder, embeddings, phase_train_placeholder)
        print("The num of DB_vector", len(DB_vector))
        #DB_vector = []
        #   启动摄像头
        Cam(DB_vector, DB_category_list, sess, images_placeholder, embeddings, phase_train_placeholder, pnet, rnet, onet )
        pass
    pass
    pass


if __name__ == '__main__':
    main()