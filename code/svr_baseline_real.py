# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 16:57:39 2021

@author: Admin
"""

import os
import scipy.io as sio
import numpy as np
from liblinear.liblinearutil import *
import csv
import argparse
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
from tkinter import _flatten

def multipl(a, b):
    sumofab = 0.0
    for i in range(len(a)):
        temp = a[i] * b[i]
        sumofab += temp
    return sumofab


def corrcoef(x, y):
    n = len(x)
    # 求和
    sum1 = sum(x)
    sum2 = sum(y)
    # 求乘积之和
    sumofxy = multipl(x, y)
    # 求平方和
    sumofx2 = sum([pow(i, 2) for i in x])
    sumofy2 = sum([pow(j, 2) for j in y])
    num = sumofxy - (float(sum1) * float(sum2) / n)
    # 计算皮尔逊相关系数
    den = sqrt((sumofx2 - float(sum1 ** 2) / n) * (sumofy2 - float(sum2 ** 2) / n))
    return num / den


def load_data(data_dir, label_dir):
    all_data  = sio.loadmat(data_dir)['feature_map']
    all_label = sio.loadmat(label_dir)['perclos']
    return all_data, all_label


def main():
    parser = argparse.ArgumentParser(description='real or lab, 0 for real, 1 for lab')
    parser.add_argument('-real_lab', type=int, default=0)
    args     = parser.parse_args()
    real_lab = args.real_lab
    
    if real_lab == 0:
        feature_dir   = 'E:/学校/大三暑期实习/dataset/datasetv6/real/egg/'
        label_dir     = 'E:/学校/大三暑期实习/dataset/datasetv6/real/eye/'  
    else:
        feature_dir   = 'E:/学校/大三暑期实习/dataset/datasetv6/lab/egg/'
        label_dir     = 'E:/学校/大三暑期实习/dataset/datasetv6/lab/eye/'  
    fold_num   = 5
    
    all_person_name = os.listdir(feature_dir)
    all_person_name.sort()
    for i in range(len(all_person_name)):
        cur_feature_path = feature_dir + all_person_name[i]
        cur_label_path   = label_dir + all_person_name[i][:-12] + '.mat'
        cur_data     = sio.loadmat(cur_feature_path)['de_feature_smooth']    
        cur_label        = sio.loadmat(cur_label_path)['perclos']
        
        # if np.abs(len(cur_feature)-len(cur_label)) > 10:
        #     if (len(cur_feature)-len(cur_label)) < 0:
        #         print('more eye move')
        #     print(np.abs(len(cur_feature)-len(cur_label)))
        #     print(all_person_name[i])
        used_len      = np.min((len(cur_data), len(cur_label)))
        used_data     = cur_data[3:used_len, :]
        used_label    = cur_label[3:used_len, :]
        each_fold_len = len(used_data)//fold_num
        used_len      = each_fold_len * fold_num
        used_data     = cur_data[:used_len, :]
        used_label    = cur_label[:used_len, :]        
        all_p_label   = []
        for j in range(fold_num):
            all_idx       = [k for k in range(len(used_data))]
            test_idx      = [k for k in range(each_fold_len*j, each_fold_len*(j+1))]
            for k in range(len(test_idx)):
                all_idx.remove(test_idx[k])
            train_idx     = all_idx
            train_data    = used_data[train_idx]
            test_data     = used_data[test_idx]
            train_label   = used_label[train_idx]
            test_label    = used_label[test_idx]            

            tmp = np.concatenate((train_data, train_label), axis = 1)
            b = np.random.shuffle(tmp)

            train_data = tmp[:, :90]
            train_lable = tmp[:,90]
            
            print(tmp.shape)
            print(train_data.shape)
            print(train_label.shape)
            
            cc = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            train_new       = train_data.tolist()
            test_new        = test_data.tolist()
            train_label_new = np.reshape(train_label,(len(train_label),)).tolist()
            test_label_new  = np.reshape(test_label,(len(test_label),)).tolist()  
            max_corr = -1
            for c in cc:
                para ='-s 11 -c %f -q'%(2**c)
                model = train(train_label_new, train_new, para)
                p_label, p_acc, p_val = predict(test_label_new, test_new, model)
                coeff = corrcoef(test_label, p_label)
                if coeff > max_corr:
                    max_corr    = coeff
                    max_p_label = p_label
            # print(max_corr)
            all_p_label.append(max_p_label)
        
        all_p_label = _flatten(all_p_label)
        coeff = corrcoef(used_label, all_p_label)     
        all_acc = []
        all_acc.append(all_person_name[i])
        all_acc.append(coeff)
        with open("result_real_5.csv","a", newline='') as datacsv:
            csvwriter = csv.writer(datacsv,dialect=("excel"))
            csvwriter.writerow(all_acc)
            
    
if __name__ == '__main__':
    main()