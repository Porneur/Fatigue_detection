#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 08:55:43 2021

@author: goumingyu
"""
from signaltools import *
import matplotlib.pyplot as plt
import os 
root =  '/home/goumingyu/document/fatigue_detection/data/real/extracted'
a = os.listdir(root)

result = []
stat = []

for i in range(len(a)):
    temp = load(root+"/"+a[i])
    result.extend(temp)
    del temp
    
for i in range(len(result)):
    result[i][1] = round(result[i][1],2)
    stat.append(result[i][1])
    
    
plt.hist(stat, rwidth=0.5, bins = 100)
plt.show()

output = []
counter = [0 for i in range(101)] 
for i in range(len(result)):
    if counter[int(result[i][1]*100)] < 200:
        output.append(result[i])
        counter[int(result[i][1]*100)] = counter[int(result[i][1]*100)]+1
 
    

save(output,"output")
new =load("output.pickle")