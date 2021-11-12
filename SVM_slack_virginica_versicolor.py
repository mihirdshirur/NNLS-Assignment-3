import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from qpsolvers import quadprog_solve_qp

df = pd.read_csv("/Users/apple/Desktop/Coding/Sem5/Assignment3_Code/iris.data")
x = []
error1 = []
data_setosa = df.iloc[0:50,0:4].values
data_versicolor = df.iloc[50:100,0:4].values
data_virginica = df.iloc[100:150,0:4].values

# Setosa is of class 1, versicolor is of class 2, virginica is of class 3
data_class_virginica = np.concatenate((np.ones((50,1)),data_virginica,np.ones((50,1))*(-1)),axis = 1)
data_class_versicolor = np.concatenate((np.ones((50,1)),data_versicolor,np.ones((50,1))),axis = 1)

data_class = np.concatenate((data_class_virginica,data_class_versicolor),axis = 0)
np.random.shuffle(data_class)
percentage = 0.7                # Percentage of training set
size = 70
training_set = data_class[0:int(percentage*100),:]       # Training data
test_set = data_class[int(percentage*100):100,:]         # Test data

P = np.zeros((75,75))
for i in range(75):
    for j in range(75):
        if i == j and i ==0:
            P[i][j] = 10**(-10)
        if i == j and i>=5:
            P[i][j] = 10**(-10)
        if i == j and i<5 and i!=0:
            P[i][j] = 1
G = np.zeros((70,75))
h = np.zeros(70)
q = np.zeros(75)
for i in range(75):
    if i>=5:
        q[i] = 10         # constant C
A = np.zeros((70,75))
b = np.zeros(70)
lb = 0
ub = np.Inf
for i in range(70):
    for j in range(5):
        G[i][j] = -training_set[i,j]*training_set[i,5]
    G[i][i+5] = 1/training_set[i,5]
for i in range(70):
    h[i] = -1

weights = quadprog_solve_qp(P, q, G, h, A , b, lb, ub)



err = 0



  
for i in range(100-size):
    predict = np.dot(weights[0:5],np.transpose(test_set[i,0:5]))
    
    if test_set[i,5] == 1 and predict<-1:
        err = err + 1
    elif test_set[i,5] == -1 and predict>1:
        err = err + 1

