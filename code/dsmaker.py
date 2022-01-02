# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 16:11:19 2021

@author: 82010
"""

import h5py
import numpy as np
import pandas as pd

import copy

import mne 
import signaltools
import os
import re
import scipy.io as scio
import tqdm

channels = ['EEG P3 - Pz', 'EEG P4 - Pz', 'EEG O1 - Pz', 'EEG O2 - Pz']

dataset_lab = []
matlist = []
#dataFile0 = 'test\\lab\\hexingtao_20190418_lab.mat'
#dataFile1 = 'test\\lab\\hexingtao_20190610_lab.mat'
dataFile2 = 'test\\real\\hexingtao_20190514_real.mat'
#data0 = scio.loadmat(dataFile0)['perclos']
#data1 = scio.loadmat(dataFile1)['perclos']
data2 = scio.loadmat(dataFile2)['perclos']
#matlist.append(data0)
#matlist.append(data1)
matlist.append(data2)


raw_path0 = 'test\\lab\\hexingtao_20190418_lab_raw.edf'
raw_path1 = 'test\\lab\\hexingtao_20190610_lab_raw.edf'
raw_path2 = 'test\\real\\hexingtao_20190514_real_raw.edf'

edflist = []
#edflist.append(mne.io.read_raw_edf(raw_path0))
#edflist.append(mne.io.read_raw_edf(raw_path1))
edflist.append(mne.io.read_raw_edf(raw_path2))


for k in range(len(edflist)):
    fs = int(edflist[k].info['sfreq'])
    segmentlen = matlist[k].shape[0]
    xs = edflist[k][channels][0]
    t = edflist[k][channels][1]
    #x_new = []
    for i in range(len(xs)):
        y = fda(xs[i], 1, 40, fs)
        y = 100000*y
        xs[i] = y
        #x_new.append(x)
        del y
        
    #xs = x_new
    #x_new = []
    #for i in tqdm.trange(11):
    for i in tqdm.trange(segmentlen):
        ttemp = edflist[k][channels][1][fs*(8*i+5):fs*(8*i+15)]
        tsegdata = []
        for x in xs:
            ytemp = x[fs*(8*i+5):fs*(8*i+15)]
            spect = wvletspect(ytemp, ttemp, fs)[:,300:-300]
            de = diffentropy(ytemp[fs:-fs], fs)
            tsegdata.append(spect)
            tsegdata.append(de)
        del ytemp
        del spect
        del de
        dataset_lab.append([tsegdata,matlist[k][i][0]])
        if ((i+1)%10 == 0):  #save period is 5
            #f = h5py.File("mytestfile.hdf5", "a")
            av = []
            for j in range(10):
                temp = []
                temp = dataset_lab[j][0]
                temp1 = dataset_lab[j][1]
                temp.append(temp1)
                av.append(temp)
            a = pd.DataFrame(
                av,     
                index=[i-x for x in range(10)],
                columns=['P3SG','P3DE','P4SG','P4DE','O1SG','O1DE','O2SG','O2DE','perclos']
            )
            del av
            a.to_pickle('ds.pickle')
            del dataset_lab
            del a
            dataset_lab = []
        del ttemp
        del tsegdata
            #PERCLOS = []
            #P3PZ = []
            #for elem in dataset_lab:
            #    PERCLOS.append(elem[1])
            #    P3PZ.append(elem[0][0])
            #dset = f.create_dataset(name = "PERCLOS", data = PERCLOS)
            #dset = f.create_dataset(name = "EEG P3 - Pz", data = P3PZ)
            #f.close()
            #dataset_lab = []
            #continue
    del xs
    del t
            
    
#save(dataset_lab,'testset')

            
            
            
        #ytemp = y[int(i*fs*8):int((i+1)*fs*8)]
        #ttemp = t[int(i*fs*8):int((i+1)*fs*8)]
        #spect = wvletspect(ytemp, ttemp, fs)
        #de = diffentropy(ytemp, fs)
        #dataset_lab.append([spect,de,matlist[index][i][0]])
    

