# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 16:49:12 2021

@author: 82010
"""


def save(alist, name):
    f = open(str(name)+'.pickle','wb')
    pickle.dump(alist, f)
    f.close()
    
def load(path):
    temp = []
    list_file = open(path,'rb')
    temp = pickle.load(list_file)
    return temp

import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVR
from sklearn.metrics import r2_score


data = load('testset.pickle')


#用迭代的方法打开追加存储的pickle
'''
with open('test','rb') as f:
    while True:
        try:
            aa=pickle.load(f)
            print(aa)
        except EOFError:
            break
'''




xt = []
x = []
y = []
for i in range(len(data)):
    y.append(data[i][1])
    x1 = (data[i][0][0] + data[i][0][2] + data[i][0][4] + data[i][0][6]).reshape(-1)
    x2 = (data[i][0][1] + data[i][0][3] + data[i][0][5] + data[i][0][7])
    x.append(np.append(x1,x2))
    xt.append(x1)
    
import sklearn.decomposition as sk_decomposition
pca = sk_decomposition.PCA(n_components= 10 ,whiten=False,svd_solver='auto')
pca.fit(xt)
reduced_X = pca.transform(xt) #reduced_X为降维后的数据
print('PCA:')
print ('降维后的各主成分的方差值占总方差值的比例',pca.explained_variance_ratio_)
print ('降维后的各主成分的方差值',pca.explained_variance_)
print ('降维后的特征数',pca.n_components_)



x = []
for i in range(len(reduced_X)):
    x1 = reduced_X[i]
    x2 = data[i][0][1] + data[i][0][3] + data[i][0][5] + data[i][0][7]
    x.append(np.append(x1,x2))

x = np.array(x)
y = np.array(y)

# fig,ax = plt.subplots(figsize=(8,6))
# ax.scatter(y, x[:, 0], s=30, c='b', marker='o')
# ax.scatter(y, x[:, 1], s=30, c='c', marker='^')
# plt.show()
# exit(0)

clf = SVR(kernel='rbf', C=1.25)
x_tran,x_test,y_train,y_test = train_test_split(x, y, test_size=0.25)
clf.fit(x_tran, y_train)
y_hat = clf.predict(x_test)

print("得分:", r2_score(y_test, y_hat))

r = len(x_test) + 1
print(y_test)
plt.plot(np.arange(1,r), y_hat, 'go-', label="predict")
plt.plot(np.arange(1,r), y_test, 'co-', label="real")
plt.legend()
plt.show()

'''
scatter(x, y, s=None, c=None, marker=None, cmap=None, norm=None, 
	vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, 
	edgecolors=None, hold=None, data=None, **kwargs)
x，y：输入数据，array_like，shape（n，）
s: 点的大小
c: 点的颜色 b-bule、c-cyan、g-green、k-black、m-magenta、r-red、w-white、y-yellow
marker：点的形状
alpha：透明度
label: 点标记

plt.plot(x, y, format_string, **kwargs)
x,y: x、y轴数据，列表或数组
format_string: 'go-'; g颜色和scatter c参数相同之处rgb颜色、o标记类型、-线的类型
**kwargs:
	color: 控制颜色, color='green'
	linestyle : 线条风格, linestyle='dashed'

'''