#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 14:59:03 2021

@author: goumingyu
"""

from tensorflow.keras import Input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
import tensorflow as tf
from tensorflow.keras.layers import Dense, AlphaDropout, GlobalMaxPooling2D, Dropout, BatchNormalization, Conv1D
from tensorflow.keras import Model
from sklearn import datasets
import numpy as np
import time 
from PIL import Image
import pickle
import os
import random
import tqdm
import signaltools

def save(alist, name):
    f = open(str(name)+'.pickle','wb')
    pickle.dump(alist, f)
    f.close()
    
def load(path):
    temp = []
    list_file = open(path,'rb')
    temp = pickle.load(list_file)
    return temp


def ReadTxtName(rootdir):
    lines = []
    with open(rootdir, 'r') as file_to_read:
        while True:
            line = file_to_read.readline()
            if not line:
                break
            line = line.strip('\n')
            lines.append(line)
    return lines


def fix(y_hat):
    temp = y_hat.T
    temp = temp[0]
    for i in range(len(temp)):
        if (temp[i] < 0) or (temp[i] > 1):
            try:
                temp[i] = temp[i-1]
            except:
                continue
        else:
            continue
    return temp


def plot_result(y, y_hat):
    n = len(y)
    coef = np.corrcoef(y,y_hat)
    t = np.linspace(0*8,(n-1)*8,n)
    plt.title('Perclos')
    plt.plot(t, y,  color='skyblue', label='real_perclos')
    plt.plot(t, y_hat, color='red', label='predict_perclos')
    plt.legend()
    plt.xlabel('time(second)')
    plt.ylabel('perclos')
    plt.text(0.5, 0.5, "CORRELATION COEF IS " + str(coef[0][1]))
    plt.show()
    



class data_chooser:
    def __init__(self, filepath):
        self.filelist = ReadTxtName(filepath)
    def length(self):
        return len(self.filelist)
    def pick(self):
        self.number = len(self.filelist)
        index = random.randint(0, self.number-1)
        data = load(self.filelist[index])
        images_train = []
        images_train2 = []
        images_train3 = []
        images_train4 = []
        lables_train= []
        DE = []
        for i in range(len(data)):
            channel1 = data[i][0][0]
            channel2 = data[i][0][2]
            channel3 = data[i][0][4]
            channel4 = data[i][0][6]
            lables_train.append(data[i][1])
            im=Image.fromarray(channel1)
            img = im.convert("RGB")
            img = np.asarray(img)
            images_train.append(img)
            del channel1
            im=Image.fromarray(channel2)
            img = im.convert("RGB")
            img = np.asarray(img)
            images_train2.append(img)
            del channel2
            im=Image.fromarray(channel3)
            img = im.convert("RGB")
            img = np.asarray(img)
            images_train3.append(img)
            del channel3
            im=Image.fromarray(channel4)
            img = im.convert("RGB")
            img = np.asarray(img)
            images_train4.append(img)
            del channel4
            DE.append([data[i][0][1],data[i][0][3],data[i][0][5],data[i][0][7]])
        images_train = np.float64(images_train)
        images_train2 = np.float64(images_train2)
        images_train3 = np.float64(images_train3)
        images_train4 = np.float64(images_train4)
        DE = np.float64(DE)
        images_train = preprocess_input(images_train)
        images_train2 = preprocess_input(images_train2)
        images_train3 = preprocess_input(images_train3)
        images_train4 = preprocess_input(images_train4)
        lables_train = np.float64(lables_train)
        return [images_train, images_train2, images_train3, images_train4, DE], lables_train
    def reduced_pick(self, a = None, b = None):
        self.number = len(self.filelist)
        index = random.randint(0, self.number-1)
        data = load(self.filelist[index])
        #data = load(self.filelist[0])
        images_train = []
        images_train2 = []
        images_train3 = []
        images_train4 = []
        lables_train= []
        DE = []
        #a = random.randint(0, int(2*len(data)/3))
        #b = a + int(len(data)/4)
        #a = 0
        #b = len(data)
        for i in range(a,b):
            channel1 = data[i][0][0]
            channel2 = data[i][0][2]
            channel3 = data[i][0][4]
            channel4 = data[i][0][6]
            lables_train.append(data[i][1])
            im=Image.fromarray(channel1)
            img = im.convert("RGB")
            img = np.asarray(img)
            images_train.append(img)
            del channel1
            im=Image.fromarray(channel2)
            img = im.convert("RGB")
            img = np.asarray(img)
            images_train2.append(img)
            del channel2
            im=Image.fromarray(channel3)
            img = im.convert("RGB")
            img = np.asarray(img)
            images_train3.append(img)
            del channel3
            im=Image.fromarray(channel4)
            img = im.convert("RGB")
            img = np.asarray(img)
            images_train4.append(img)
            del channel4
            DE.append([data[i][0][1],data[i][0][3],data[i][0][5],data[i][0][7]])
        images_train = np.float64(images_train)
        images_train2 = np.float64(images_train2)
        images_train3 = np.float64(images_train3)
        images_train4 = np.float64(images_train4)
        DE = np.float64(DE)
        images_train = preprocess_input(images_train)
        images_train2 = preprocess_input(images_train2)
        images_train3 = preprocess_input(images_train3)
        images_train4 = preprocess_input(images_train4)
        lables_train = np.float64(lables_train)
        return [images_train[:-1], images_train2[:-1], images_train3[:-1], images_train4[:-1], DE[:-1], lables_train[:-1],
                images_train[1:], images_train2[1:], images_train3[1:], images_train4[1:], DE[1:]], lables_train[1:]









filepath = "/home/goumingyu/document/fatigue_detection/path/testpath.txt"
restored_model = tf.keras.models.load_model('/home/goumingyu/document/fatigue_detection/result/model/path_to_saved_alter_model')
generator = data_chooser(filepath)
x,y = generator.reduced_pick(100,120)
y_hat = restored_model.predict(x)
y_hat = fix(y_hat)
plot_result(y, y_hat)







