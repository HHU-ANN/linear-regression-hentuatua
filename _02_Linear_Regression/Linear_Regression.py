# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np


def ridge(data):
    X,y = read_data()
    tem = X [:,0] * X [:,1]
    X = np.column_stack((X,tem))
    tem = X [:,2] * X [:,3]
    X = np.column_stack((X,tem))
    tem = X [:,4] * X [:,5]
    X = np.column_stack((X,tem))
    b = np.ones(X.shape[0])
    X = np.column_stack((X,b))
    l = 0.2
    I = np.eye(10)
    data = np.append(data,[data[0]*data[1],data[2]*data[3],data[4]*data[5],1])
    weight = np.matmul(np.linalg.inv(np.matmul(X.T,X)+ l*I),np.matmul (X.T,y))
    return weight @ data
    
def lasso(data):
    X,y = read_data()
    y =np.mat(y.reshape(-1,1))
    tem = X [:,0] * X [:,1]
    X = np.column_stack((X,tem))
    tem = X [:,2] * X [:,3]
    X = np.column_stack((X,tem))
    tem = X [:,4] * X [:,5]
    X = np.column_stack((X,tem))
    b = np.ones(X.shape[0])
    X = np.column_stack((X,b))
    m = X.shape[0]
    n = X.shape[1]
    l = 0.02
    a = 2e-14
    t = 10000
    w = np.zeros(n)
    #w=np.mat(w)
    w=w.reshape(-1,1)
    data = np.append(data,[data[0]*data[1],data[2]*data[3],data[4]*data[5],1])
    for i in range (t):
        gra= (np.matmul(X.T,np.matmul(X,w)-y) /n) + l * np.sign(w)
        w = w - a * gra
    return data @ w



def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y


