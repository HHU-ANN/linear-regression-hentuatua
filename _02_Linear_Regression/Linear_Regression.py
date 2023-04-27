# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np


def ridge(data):
    X,y = read_data()
    b = np.ones(X.shape[0])
    X = np.column_stack((X,b))
    l = 0.5
    I = np.eye(7)
    weight = np.matmul(np.linalg.inv(np.matmul(X.T,X)+np.matmul(l,I)),np.matmul (X.T,y))
    return weight @ data
    
def lasso(data):
    X,y = read_data()
    b = np.ones(X.shape[0])
    X = np.column_stack((X,b))
    l = 0.5
    a = 0.5
    t = 100
    w = np.zeros(7)
    for i in range (t):
        s = np.zeros(7)
        for j in range (X.shape[0]-1):
            s = s - np.matmul(X,y-np.matmul(X,w)) + l * sgn(w)
        w = w - a / X.shape[0] * s


def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y


