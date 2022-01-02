#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 16:53:51 2021

@author: goumingyu
"""

import pandas as pd
from model_lstm import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import re
from tensorflow.keras.utils import plot_model



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


def CCC(x,y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    variance_x = np.var(x)
    variance_y = np.var(y)
    cov = np.cov(x,y, bias = True)[0][1]
    return (2*cov)/(variance_x + variance_y + (mean_x-mean_y)**2)

            

def new_validate(x_path, y_path, model):
    x_raw_1 = scio.loadmat(x_path)['de_feature_smooth']
    x_raw_2 = scio.loadmat(x_path)['psd_feature_smooth']
    x_raw = np.hstack((x_raw_1, x_raw_2))
    y_raw = scio.loadmat(y_path)['perclos']
    reduced_shape = min(x_raw.shape[0], y_raw.shape[0])
    y_raw = y_raw[:reduced_shape,]
    x_raw = x_raw[:reduced_shape,]
    dataset = np.hstack((x_raw, y_raw))
    TRAIN_SPLIT = dataset.shape[0] - 1
    data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
    data_std = dataset[:TRAIN_SPLIT].std(axis=0)
    dataset = (dataset-data_mean)/data_std
    past_history = 10   #用过去10个单元的数据
    future_target = 0   #预测未来1个单元的数据
    STEP = 1

    x_single, y_single = multivariate_data(dataset, y_raw, 0,
                                                       TRAIN_SPLIT, past_history,
                                                       future_target, STEP,
                                                       single_step=True)
    for i in range(len(x_single)):
        x_single[i][9][180] = 0
    
    hat = model.predict(x_single)
    length = len(hat[:,0])
    t = [i*8 for i in range(length)]
    plt.plot(t,hat[:,0])
    coef = CCC(y_single[:,-1],hat[:,0])
    ret = re.findall(r"egg/(.*)_raw",x_path)
    temp = ret[0]
    plt.show()
    plt.close()
    #model.save("/home/goumingyu/document/fatigue_detection/result/model/real_models/"+temp, save_format='tf')
    return coef


def validate(x_path, y_path, model):
    x_raw_1 = scio.loadmat(x_path)['de_feature_smooth']
    x_raw_2 = scio.loadmat(x_path)['psd_feature_smooth']
    x_raw = np.hstack((x_raw_1, x_raw_2))
    y_raw = scio.loadmat(y_path)['perclos']
    reduced_shape = min(x_raw.shape[0], y_raw.shape[0])
    y_raw = y_raw[:reduced_shape,]
    x_raw = x_raw[:reduced_shape,]
    dataset = np.hstack((x_raw, y_raw))
    TRAIN_SPLIT = dataset.shape[0] - 1
    data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
    data_std = dataset[:TRAIN_SPLIT].std(axis=0)
    dataset = (dataset-data_mean)/data_std
    past_history = 10   #用过去10个单元的数据
    future_target = 0   #预测未来1个单元的数据
    STEP = 1

    x_single, y_single = multivariate_data(dataset, y_raw, 0,
                                                       TRAIN_SPLIT, past_history,
                                                       future_target, STEP,
                                                       single_step=True)
    for i in range(len(x_single)):
        x_single[i][9][180] = 0
    
    hat = model.predict(x_single)
    length = len(hat[:,0])
    t = [i*8 for i in range(length)]
    plt.plot(t,hat[:,0])
    coef = np.corrcoef(y_single[:,-1],hat[:,0])
    plt.plot(t,y_single[:,-1])
    plt.legend(["predict perclos", "real perclos"])
    plt.xlabel('time in second')
    plt.ylabel('Perclos')
    plt.text(100,1,"CORRELATION COEF IS " + str(coef[0][1]))
    ret = re.findall(r"egg/(.*)_raw",x_path)
    temp = ret[0]
    plt.savefig("/home/goumingyu/document/fatigue_detection/result/figure/lstm_ltrt_savefig/" + temp + ".jpg")
    plt.close()
    return coef[0][1]


def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i+target_size-1])
        else:
            labels.append(target[i:i+target_size-1])

    return np.array(data), np.array(labels)


def get_unit_length(df):
    #get the absolute period of each unit(s), ex: 900 presents 15 minutes. 
    features.index = df['time']
    unit_length = features.index[1]-features.index[0]
    return unit_length

'''
def mat_to_dataset(x_path, y_path):
    x_raw_1 = scio.loadmat(x_path)['de_feature_smooth']
    x_raw_2 = scio.loadmat(x_path)['psd_feature_smooth']
    x_raw = np.hstack((x_raw_1, x_raw_2))
    y_raw = scio.loadmat(y_path)['perclos']
    reduced_shape = min(x_raw.shape[0], y_raw.shape[0])
    y_raw = y_raw[-reduced_shape:,]
    x_raw = x_raw[-reduced_shape:,]
    TRAIN_SPLIT = x_raw.shape[0]-1
    data_mean = x_raw[:TRAIN_SPLIT].mean(axis=0)
    data_std = x_raw[:TRAIN_SPLIT].std(axis=0)
    x_raw = (x_raw-data_mean)/data_std
    dataset = np.hstack((x_raw, y_raw))
    past_history = 10   #用过去10个单元的数据
    future_target = 1   #预测未来1个单元的数据
    STEP = 1

    x_train_single, y_train_single = multivariate_data(dataset, y_raw, 0,
                                                       TRAIN_SPLIT, past_history,
                                                       future_target, STEP,
                                                       single_step=True)
    return x_train_single, y_train_single
''' 

def mat_to_dataset(x_path, y_path):
    x_raw_1 = scio.loadmat(x_path)['de_feature_smooth']
    x_raw_2 = scio.loadmat(x_path)['psd_feature_smooth']
    x_raw = np.hstack((x_raw_1, x_raw_2))
    y_raw = scio.loadmat(y_path)['perclos']
    reduced_shape = min(x_raw.shape[0], y_raw.shape[0])
    y_raw = y_raw[-reduced_shape:,]
    x_raw = x_raw[-reduced_shape:,]
    dataset = np.hstack((x_raw, y_raw))
    TRAIN_SPLIT = x_raw.shape[0]-1
    data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
    data_std = dataset[:TRAIN_SPLIT].std(axis=0)
    dataset = (dataset-data_mean)/data_std
    past_history = 10   #用过去10个单元的数据
    future_target = 0   #预测未来1个单元的数据
    STEP = 1

    x_train_single, y_train_single = multivariate_data(dataset, y_raw, 0,
                                                       TRAIN_SPLIT, past_history,
                                                       future_target, STEP,
                                                       single_step=True)
    for i in range(len(x_train_single)):
        x_train_single[i][9][180] = 0
    return x_train_single, y_train_single   
    
x_real_paths = ReadTxtName('/home/goumingyu/document/fatigue_detection/path/v6_real_egg_path.txt')
y_real_paths = ReadTxtName('/home/goumingyu/document/fatigue_detection/path/v6_real_eye_path.txt')
x_lab_paths = ReadTxtName('/home/goumingyu/document/fatigue_detection/path/v6_lab_egg_path.txt')
y_lab_paths = ReadTxtName('/home/goumingyu/document/fatigue_detection/path/v6_lab_eye_path.txt')

def train_except_k(k, x_paths, y_paths):
    started = 0
    for i in range(len(x_paths)):
        if (i != k ):
            temp1, temp2 = mat_to_dataset(x_paths[i], y_paths[i])
            if (started == 0):
                x_train_single = temp1
                y_train_single = temp2
                started = 1
            else:
                 x_train_single = np.vstack((x_train_single, temp1))
                 y_train_single = np.vstack((y_train_single, temp2))
                 
    x_val_single, y_val_single = mat_to_dataset(x_paths[k], y_paths[k])
    
    BUFFER_SIZE = 256
    BATCH_SIZE = 64
    train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
    train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()


    val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
    val_data_single = val_data_single.batch(BATCH_SIZE).repeat()

    model = RollModel(x_train_single.shape[-2:])
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')
    EPOCHS = 1000
    EVALUATION_INTERVAL = 100
    model_history = model.fit(train_data_single, epochs=EPOCHS,
                                                steps_per_epoch=EVALUATION_INTERVAL,
                                                validation_data=val_data_single,
                                                validation_steps=10)
    
    #coef = validate(x_paths[k], y_paths[k], model)
    coef = new_validate(x_paths[k], y_paths[k], model)
    return coef
                
                
def lab_train_real_test(x_real_paths, y_real_paths, x_lab_paths, y_lab_paths):
    started = 0
    for i in range(len(x_lab_paths)):
        temp1, temp2 = mat_to_dataset(x_lab_paths[i], y_lab_paths[i])
        if (started == 0):
            x_train_single = temp1
            y_train_single = temp2
            started = 1
        else:
            x_train_single = np.vstack((x_train_single, temp1))
            y_train_single = np.vstack((y_train_single, temp2))
                 
    
    
    started = 0
    for i in range(len(x_real_paths)):
        temp1, temp2 = mat_to_dataset(x_real_paths[i], y_real_paths[i])
        if (started == 0):
            x_val_single = temp1
            y_val_single = temp2
            started = 1
        else:
            x_val_single = np.vstack((x_val_single, temp1))
            y_val_single = np.vstack((y_val_single, temp2))
    
    
    BUFFER_SIZE = 256
    BATCH_SIZE = 64
    train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
    train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    
    val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
    val_data_single = val_data_single.batch(BATCH_SIZE).repeat()
    print(x_train_single.shape[-2:])

    model = RollModel(x_train_single.shape[-2:])
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')
    EPOCHS = 1
    EVALUATION_INTERVAL = 100
    model_history = model.fit(train_data_single, epochs=EPOCHS,
                                                steps_per_epoch=EVALUATION_INTERVAL,
                                                validation_data=val_data_single,
                                                validation_steps=10)
    
    return model


            
    



'''
for i in range(len(x_paths)-1):
    temp1, temp2 = mat_to_dataset(x_paths[i], y_paths[i])
    if (i == 0):
        x_train_single = temp1
        y_train_single = temp2
    else:
        x_train_single = np.vstack((x_train_single, temp1))
        y_train_single = np.vstack((y_train_single, temp2))

x_val_single, y_val_single = mat_to_dataset(x_paths[-1], y_paths[-1])

    
BUFFER_SIZE = 256
BATCH_SIZE = 64
train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()


val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
val_data_single = val_data_single.batch(BATCH_SIZE).repeat()

model = RollModel(x_train_single.shape[-2:])
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')
EPOCHS = 1000
EVALUATION_INTERVAL = 100
model_history = model.fit(train_data_single, epochs=EPOCHS,
                                            steps_per_epoch=EVALUATION_INTERVAL,
                                            validation_data=val_data_single,
                                            validation_steps=10)
'''
'''
result = []
for k in range(len(x_lab_paths)):
    coef = train_except_k(k, x_real_paths, y_real_paths)
    result.append(coef)
'''


result = []
model = lab_train_real_test(x_real_paths, y_real_paths, x_lab_paths, y_lab_paths)

for i in range(len(x_real_paths)):
    coef = new_validate(x_real_paths[i], y_real_paths[i], model)
    result.append(coef)

#restored_model = tf.keras.models.load_model('/home/goumingyu/document/fatigue_detection/code/saved_lstm_model')

#model.save('saved_lstm_model', save_format='tf')




