from scipy import misc
import os
import threading
import matplotlib.pyplot as plt     # plt 用于显示图片
import matplotlib.image as mpimg
import time
from PIL import Image
#coding: utf-8
class ThreadClass(threading.Thread):
    def __init__(self, path, name):
        threading.Thread.__init__(self)
        self.path = path
        self.name = name
    def run(self):
        lena = mpimg.imread(self.path)
        plt.imshow(lena)  # 显示图片
        plt.title(label=(self.name + "\nInformation:  - - - \nAge:  - - -\nSex:   - - -"))
        plt.axis('off')  # 不显示坐标轴
        plt.show()
        time.sleep(5)
        #im = Image.open(self.path)
        #im.show()


def ShowInfo(name):
    path = os.path.join("./face/DB", name)
    for _, _, files in os.walk(path):
        file = files[0]
        image_path = os.path.join(path, file)
        try:
            temp = ThreadClass(image_path, name)
            temp.start()
        except:
            pass
        pass
    pass


def main():
    ShowInfo("Chen_Bin")
    pass

if __name__ == '__main__':
    main()