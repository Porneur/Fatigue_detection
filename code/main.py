import mne 
from signaltools import *
import os
import re
import scipy.io as scio
import gc
import tqdm


channels = ['EEG P3 - Pz', 'EEG P4 - Pz', 'EEG O1 - Pz', 'EEG O2 - Pz']
edf_lab = ReadTxtName("eegpath.txt")
mat_lab = ReadTxtName("eyepath.txt")


def feature_extract(edflist, matlist):
    dataset_lab = []
    edflist = mne.io.read_raw_edf(edf_lab[k])
    matlist = scio.loadmat(mat_lab[k])['perclos']
    fs = int(edflist.info['sfreq'])
    segmentlen = matlist.shape[0]
    xs = edflist[channels][0]
    t = edflist[channels][1]
    #x_new = []
    for i in range(len(xs)):
        y = fda(xs[i], 1, 40, fs)
        y = 100000*y
        xs[i] = y
        #x_new.append(x)
        
    #xs = x_new
    #x_new = []
    #for i in tqdm.trange(1):
    for i in tqdm.trange(segmentlen):
        try:
            ttemp = edflist[channels][1][fs*(8*i+5):fs*(8*i+15)]
            tsegdata = []
            for x in xs:
                ytemp = x[fs*(8*i+5):fs*(8*i+15)]
                #spect = reduce_size(wvletspect(ytemp, ttemp, fs)[:,300:-300], 240, 88)
                spect = wvletspect(ytemp, ttemp, fs)
                de = diffentropy(ytemp[fs:-fs], fs)
                tsegdata.append(spect)
                tsegdata.append(de)
                del spect
                del ytemp
                del de
            dataset_lab.append([tsegdata,matlist[i][0]])
            del tsegdata
            del ttemp
            gc.collect()
        except:
            continue
    save(dataset_lab,mat_lab[k].replace(".mat",""))
    del edflist, matlist, fs, segmentlen, xs, t, y, dataset_lab
    gc.collect()
    


for k in range(len(edf_lab)):
    edflist = mne.io.read_raw_edf(edf_lab[k])
    matlist = scio.loadmat(mat_lab[k])['perclos']
    feature_extract(edflist, matlist)
            
            
        #ytemp = y[int(i*fs*8):int((i+1)*fs*8)]
        #ttemp = t[int(i*fs*8):int((i+1)*fs*8)]
        #spect = wvletspect(ytemp, ttemp, fs)
        #de = diffentropy(ytemp, fs)
        #dataset_lab.append([spect,de,matlist[index][i][0]])
    
    
    
    
'''
    
    for channel in channels:
        x = edflist[k][channel][0][0]
        t = edflist[k][channel][1]
        y = fda(x, 1, 40, fs)
        [y, t] = Ttruncate(y, t, fs, t1 = 20, t2 = 20)
        for i in range(segmentlen):
            ytemp = y[int(i*fs*8):int((i+1)*fs*8)]
            ttemp = t[int(i*fs*8):int((i+1)*fs*8)]
            spect = wvletspect(ytemp, ttemp, fs)
            de = diffentropy(ytemp, fs)
            dataset_lab.append([spect,de,matlist[index][i][0]])
save(dataset_lab,dataset_lab)
    

'''






'''
channels = ['EEG O1 - Pz', 'EEG O2 - Pz']
'''

'''
labpath = 'lab\\egg'
realpath = 'real\\egg'
dir_list = []
labsubs = os.listdir(labpath)
realsubs = os.listdir(realpath)
expr = "[^_]*"
os.mkdir("truncated")
for elem in labsubs:
    temp = re.match(expr, elem).group(0)
    os.mkdir("truncated\\" + temp)
    os.mkdir("truncated\\" + temp + "\\lab")
for elem in realsubs:
    temp = re.match(expr, elem).group(0)
    if os.path.exists("truncated\\" + temp):
        os.mkdir("truncated\\" + temp + "\\real")
    else:
        os.mkdir("truncated\\" + temp)
        os.mkdir("truncated\\" + temp + "\\real")
 '''       
        
    




