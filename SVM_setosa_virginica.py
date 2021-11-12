import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from qpsolvers import quadprog_solve_qp

df = pd.read_csv("/Users/apple/Desktop/Coding/Sem5/Assignment3_Code/iris.data")

error1 = []
x = []
data_setosa = df.iloc[0:50,0:4].values
data_versicolor = df.iloc[50:100,0:4].values
data_virginica = df.iloc[100:150,0:4].values

# Setosa is of class 1, versicolor is of class 2, virginica is of class 3
data_class_setosa = np.concatenate((np.ones((50,1)),data_setosa,np.ones((50,1))*(-1)),axis = 1)

data_class_virginica = np.concatenate((np.ones((50,1)),data_virginica,np.ones((50,1))*1),axis = 1)
data_class = np.concatenate((data_class_setosa,data_class_virginica),axis = 0)
np.random.shuffle(data_class)
percentage = 0.7                # Percentage of training set
size = 70
training_set = data_class[0:int(percentage*100),:]       # Training data
test_set = data_class[int(percentage*100):100,:]         # Test data

P = np.array([[10**(-10),0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]],dtype='float')
G = np.zeros((size,5))
h = np.zeros(size)
q = np.zeros(5)
A = np.zeros((5,5))
b = np.zeros(5)
for i in range(size):
    for j in range(5):
        G[i][j] = -training_set[i,j]*training_set[i,5]
for i in range(70):
    h[i] = -1
weights = quadprog_solve_qp(P, q, G, h, A, b)
print("QP solution: x = {}".format(weights))


err = 0



  
for i in range(100-size):
    predict = np.dot(np.transpose(weights),np.transpose(test_set[i,0:5]))
    if test_set[i,5] == 1 and predict<-1:
        err = err + 1
    elif test_set[i,5] == -1 and predict>1:
        err = err + 1
print(err)