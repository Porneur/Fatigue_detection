#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 14:59:21 2021

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



def save(alist, name):
    f = open(str(name)+'.pickle','wb')
    pickle.dump(alist, f)
    f.close()
    
def load(path):
    temp = []
    list_file = open(path,'rb')
    temp = pickle.load(list_file)
    return temp

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
    
     
        
        









class RegModel(Model):
    def __init__(self, dim):
        super().__init__()
        #input_shape = images_train.shape
        vgg16_model1 = VGG16(weights='imagenet', include_top=False, input_shape=(dim))

        '''
        self.layer1 = VGG16(weights='imagenet', include_top=False, input_shape=(dim))
        for layer in self.layer1.layers:
            layer.trainable = False
        '''
        self.layer1 = vgg16_model1
        for layer in self.layer1.layers:
            layer.trainable = False
        self.layerA = GlobalMaxPooling2D()
        self.layerB = Dense(256, activation=tf.nn.relu,kernel_regularizer='l2')
        self.layerC = AlphaDropout(0.5)
        self.layerD = Dense(10, activation=tf.nn.relu, kernel_regularizer='l2')
        
        #self.layer21 = VGG16(weights='imagenet', include_top=False, input_shape=(dim))
        self.layer2A = GlobalMaxPooling2D()
        self.layer2B = Dense(256, activation=tf.nn.relu,kernel_regularizer='l2')
        self.layer2C = AlphaDropout(0.5)
        self.layer2D = Dense(10, activation=tf.nn.relu, kernel_regularizer='l2')
        #self.layerfin = GlobalMaxPooling2D()
        
        self.layer3A = GlobalMaxPooling2D()
        self.layer3B = Dense(256, activation=tf.nn.relu,kernel_regularizer='l2')
        self.layer3C = AlphaDropout(0.5)
        self.layer3D = Dense(10, activation=tf.nn.relu, kernel_regularizer='l2')
        
        self.layer4A = GlobalMaxPooling2D()
        self.layer4B = Dense(256, activation=tf.nn.relu,kernel_regularizer='l2')
        self.layer4C = AlphaDropout(0.5)
        self.layer4D = Dense(10, activation=tf.nn.relu, kernel_regularizer='l2')
        
        self.convlayer = Conv1D(20, 6)
        self.mixer = Dense(8, activation=tf.nn.relu, kernel_regularizer='l2')
        self.layerout = Dense(1)
        
        


    def call(self, inputs):
        x1 = inputs[0] #'EEG P3 - Pz', 'EEG P4 - Pz', 'EEG O1 - Pz' 3 channels spectrogram(similar to RGB)
        #3 Channels DE'S sum
        y = self.layer1(x1)
        y = self.layerA(y)
        y = self.layerB(y)
        y = self.layerC(y)
        y = self.layerD(y)
        
        x2 = inputs[1] #'EEG P3 - Pz', 'EEG P4 - Pz', 'EEG O1 - Pz' 3 channels spectrogram(similar to RGB)
        #3 Channels DE'S sum
        y2 = self.layer1(x2)
        y2 = self.layer2A(y2)
        y2 = self.layer2B(y2)
        y2 = self.layer2C(y2)
        y2 = self.layer2D(y2)
        
        x3 = inputs[2] #'EEG P3 - Pz', 'EEG P4 - Pz', 'EEG O1 - Pz' 3 channels spectrogram(similar to RGB)
        #3 Channels DE'S sum
        y3 = self.layer1(x3)
        y3 = self.layer3A(y3)
        y3 = self.layer3B(y3)
        y3 = self.layer3C(y3)
        y3 = self.layer3D(y3)
        
        
        x4 = inputs[3] #'EEG P3 - Pz', 'EEG P4 - Pz', 'EEG O1 - Pz' 3 channels spectrogram(similar to RGB)
        #3 Channels DE'S sum
        y4 = self.layer1(x4)
        y4 = self.layer4A(y4)
        y4 = self.layer4B(y4)
        y4 = self.layer4C(y4)
        y4 = self.layer4D(y4)
        
        z = tf.keras.layers.Concatenate()([y,y2,y3,y4])
        #concatenated_output = tf.expand_dims(concatenated_output, axis=0) 
        #z = self.convlayer(concatenated_output)
        z1 = tf.keras.layers.Concatenate()([z,inputs[4]])
        z1 = self.mixer(z1)
        z1 = self.layerout(z1)
        
        
        #temp = self.layerfin(concatenated_output)
        #z = self.layerout(concatenated_output)
        return z1
    '''
    def build_graph(self):
        x = Input(shape=(dim))
        return Model(inputs=[x], outputs=self.call(x))
    '''
    
    def build_graph(self):
        x1 = Input(shape=(dim))
        x2 = Input(shape=(dim))
        x3 = Input(shape=(dim))
        x4 = Input(shape=(dim))
        x5 = Input(shape=(4))
        return Model(inputs=[x1,x2,x3,x4,x5], outputs=self.call([x1,x2,x3,x4,x5]))


'''    
data = load('testsetlab.pickle')
images_train = []
lables_train= []
for i in range(len(data)):
    channel1 = data[i][0][0]
    #channel2 = data[i][0][2]
    #channel3 = data[i][0][4]
    #rgbArray = np.zeros((512,512,3), 'uint8')
    #rgbArray[..., 0] = r*256
    #rgbArray[..., 1] = g*256
    #rgbArray[..., 2] = b*256
    im=Image.fromarray(channel1)
    img = im.convert("RGB")
    img = np.asarray(img)
    images_train.append(img)
    lables_train.append(data[i][1])
images_train = np.float64(images_train)
lables_train = np.float64(lables_train)
images_train = preprocess_input(images_train)
'''



def build_model(dim):
    model = RegModel(dim)
    optimizer = tf.keras.optimizers.Adam(0.001)
    model.compile(loss='mse',optimizer=optimizer,metrics=['mae', 'mse'])
    return model



class PrintDot(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')



#model = RegModel((dim))
#model.build((None, *dim))


#Set training parameters
dim = (88,240,3)
Data = data_chooser('pathfile.txt')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=15, min_lr=1e-10)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
optimizer = tf.keras.optimizers.Adam(0.001)
model = RegModel(dim)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
loss = []
val_loss = []


#Start to train
for i in tqdm.trange(1,100): 
    try:
        EPOCHS = i
        images_train, lables_train = Data.pick()
        history = model.fit(
            images_train, lables_train,
            epochs=EPOCHS, validation_split = 0.2, verbose=0,
            callbacks=[PrintDot(),early_stopping,reduce_lr])
        temp_loss = history.history['loss']
        temp_val_loss = history.history['val_loss']
        loss.extend(temp_loss)
        val_loss.extend(temp_val_loss)
        del temp_loss,images_train,temp_val_loss, lables_train
    except:
        continue
    #print('loss:',loss)
    #print('val_loss:',val_loss)
    
    
#Save trained model and training metrics.    
model.save('path_to_saved_model', save_format='tf')
save(loss,"loss")
save(val_loss, "val_loss")

'''

#Show the structure of the model;
tf.keras.utils.plot_model(
    model.build_graph(),                      # here is the trick (for now)
    to_file='model.png', dpi=96,              # saving  
    show_shapes=True, show_layer_names=True,  # show shapes and layer name
    expand_nested=False                       # will show nested block
)
'''

'''
plot_model(model,to_file='model.png',show_shapes=True,show_layer_names=True,rankdir='TB')
plt.figure(figsize=(10,10))
img=plt.imread('model1.png')
plt.imshow(img)
plt.axis('off')
plt.show()

'''






















