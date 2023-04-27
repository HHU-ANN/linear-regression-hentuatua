# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np


def ridge(data):
    X,y = read_data()
    tem = X [:,1] + X [:,2]
    X = np.column_stack((X,tem))
    tem = X [:,3] - X [:,4]
    X = np.column_stack((X,tem))
    b = np.ones(X.shape[0])
    X = np.column_stack((X,b))
    l = 0.5
    I = np.eye(9)
    data = np.append(data,[data[1]+data[2],data[3]-data[4],1])
    weight = np.matmul(np.linalg.inv(np.matmul(X.T,X)+ l*I),np.matmul (X.T,y))
    return weight @ data
    
def lasso(data):
    return ridge(data)


def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y


