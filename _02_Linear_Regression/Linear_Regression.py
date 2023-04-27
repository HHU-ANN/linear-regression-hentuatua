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
    l = 0
    I = np.eye(7)
    data.append(1)
    weight = np.matmul(np.linalg.inv(np.matmul(X.T,X)+ l*I),np.matmul (X.T,y))
    return weight @ data
    
def lasso(data):
    return ridge(data)


def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y


