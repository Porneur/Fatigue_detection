from PIL import Image
import numpy as np
import scipy.signal as ss
import scipy.io as sio 
import matplotlib.pyplot as plt
import mne
import json
import os,sys
import pickle
import re
import pywt
import math
from tensorflow.keras.utils import plot_model

def Ttruncate(x, t, fs, t1 = 0, t2 = 0):  #按照时间截断数据，第三个参数代表舍去前多少秒，第四个代表舍去后多少秒
    seg1 = int(t1*fs)
    seg2 = int(t2*fs)
    if t1 != 0:
        x = x[seg1:]
        t = t[seg1:]

    if t2 != 0:
        x = x[:-seg2]    
        t = t[:-seg2]

    return x, t


'''    
from PIL import Image

width = 300
height = 88

img = a.resize((width, height),Image.ANTIALIAS)
'''


def wvletspect(x, t, fs):
    wavelet = 'cgau8'  #shan,morl and cgau8 seem to be good
    wcf = pywt.central_frequency(wavelet=wavelet)
    totalscal = 512
    cparam = 2 * wcf * totalscal
    scales = cparam/np.arange(totalscal, 1, -1)
    # 连续小波变换
    [cwtmatr, frequencies] = pywt.cwt(x, scales, wavelet, 1.0/fs)
    # 绘图
    plt.figure(figsize=(8, 4))
    plt.subplot(211)
    plt.plot(t, x)
    plt.xlabel(u"time(s)")
    plt.title(u"Time spectrum")
    plt.subplot(212)
    plt.contourf(t, frequencies[-90:-2], abs(cwtmatr)[-90:-2,:])
    plt.ylabel(u"freq(Hz)")
    plt.xlabel(u"time(s)")
    plt.subplots_adjust(hspace=0.4)
    plt.colorbar()
    plt.close()
    #return abs(cwtmatr)[-90:-2,:].astype(np.float32)
    return reduce_size(abs(cwtmatr)[-90:-2,:].astype(np.float32)[:,300:-300], 240, 88)

def reduce_size(array, width, height):
    im=Image.fromarray(array)
    im_new = im.resize((width,height))
    array_new = np.array(im_new) 
    return array_new

 

def fda(x_1,Fstop1,Fstop2,fs): #（输入的信号，截止频率下限，截止频率上限）
	b, a = ss.butter(4, [2.0*Fstop1/fs,2.0*Fstop2/fs], 'bandpass')
	filtedData = ss.filtfilt(b,a,x_1)
	return filtedData 


def find_dirs(path,directory = ''):   #传入放置所有数据的根目录
    directory = ''
    #directory = ''
    dir_list = []
    subs = os.listdir('data')
    for sub in subs:
        path_buff = []
        try:
            path_buff.append(path + '/' + sub + '/'+ sub + '.edf')
        except:
            path_buff.append(None)
        try:
            #path_buff.append(path + '/' + sub + '/'+ sub + '_spike.json')
            path_buff.append(None)
        except:
            path_buff.append(None)
        try:
            path_buff.append(path + '/' + sub + '/'+ sub + '_dataInfo.json')
            #path_buff.append(None)
        except:
            path_buff.append(None)
        try:
            path_buff.append(path + '/' + sub + '/'+ sub+ '.atn')
        except:
            path_buff.append(None)
        if path_buff[0] != None:
            dir_list.append(path_buff)
        else:
            continue
        
    return dir_list



def diffentropy(x, fs):
    [f,px]=ss.welch(x,window = ('hann'),nperseg=1024, noverlap=128,nfft=len(x),fs=fs)
    segment = [0, 0, 0, 0, 0]    #[1,4]Hz, [4,8]Hz, [8,13]Hz, [13,30]Hz, [30,50]Hz
    l0 = 0
    l1 = 0
    l2 = 0
    l3 = 0
    l4 = 0
    for i in range(len(f)):
        if (f[i]>1 and f[i] <= 4):
            segment[0] = segment[0]+px[i]
            l0 = l0 + 1
        elif (f[i]>4 and f[i] <= 8):
            segment[1] = segment[1]+px[i]
            l1 = l1 + 1
        elif (f[i]>8 and f[i] <= 13):
            segment[2] = segment[2]+px[i]
            l2 = l2 + 1
        elif (f[i]>13 and f[i] <= 30):
            segment[3] = segment[3]+px[i]
            l3 = l3 + 1
        elif (f[i]>30 and f[i] < 50):
            segment[4] = segment[4]+px[i]    
            l4 = l4 + 1
        else:
            continue
    PSD = segment[0]/l0 + segment[1]/l1 + segment[2]/l2 + segment[3]/l3 + segment[4]/l4
    DE = math.log(PSD,2)
    
    return DE
    


def save(alist, name):
    f = open(str(name)+'.pickle','ab')
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

