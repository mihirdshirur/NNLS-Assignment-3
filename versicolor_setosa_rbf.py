import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


df = pd.read_csv("/Users/apple/Desktop/Coding/Sem5/Assignment3_Code/iris.data")

error1 = []
x = []
data_setosa = df.iloc[0:50,0:4].values
data_versicolor = df.iloc[50:100,0:4].values
data_virginica = df.iloc[100:150,0:4].values

# Setosa is of class 1, versicolor is of class 2, virginica is of class 3
data_class_setosa = np.concatenate((data_setosa,np.ones((50,1))*(-1)),axis = 1)
data_class_versicolor = np.concatenate((data_versicolor,np.ones((50,1))),axis = 1)

data_class = np.concatenate((data_class_setosa,data_class_versicolor),axis = 0)
np.random.shuffle(data_class)
percentage = 0.7                # Percentage of training set
size = 70
training_set = data_class[0:int(percentage*100),:]       # Training data
test_set = data_class[int(percentage*100):100,:]         # Test data


# We implement K means clustering
K  = 10
mu = np.random.rand(K,4)          # Kx4 mean vector
for t in range(K):
    mu[t,0] = mu[t,0] + 6
    mu[t,1] = mu[t,1] + 2.5
    mu[t,2] = mu[t,2] + 4.2
    mu[t,3] = mu[t,3] + 1.5
C = np.zeros((size,1))
for i in range(100):
    # Update coder
    for j in range(size):
        dist = np.zeros((K,1))
        for k in range(K):
            dist[k,0] = np.linalg.norm(training_set[j:j+1,0:4]-mu[k:k+1,0:4])
        k_close = np.argmin(dist)
        C[j,0] = int(k_close)
    # Update means
    sum_K = np.zeros((K,4))
    count_K = np.zeros((K,1))
    for j in range(size):
        k = int(C[j,0])
        count_K[k,0] = count_K[k,0] + 1
        sum_K[k:k+1,0:4] = sum_K[k:k+1,0:4] + training_set[j:j+1,0:4]
    for k in range(K):
        if count_K[k,0] != 0:
            for t in range(4):
                mu[k,t] = sum_K[k:k+1,t:t+1]/count_K[k:k+1,0:1]
    
    
    # Calculate error for test set
    error = 0
    for j in range(100-size):
        dist = np.zeros((K,1))
        for k in range(K):
            dist[k,0] = np.linalg.norm(test_set[j:j+1,0:4]-mu[k:k+1,0:4])
        error = error + np.amin(dist)
    #print(error)

d = 0
for i in range(K):
    for j in range(K):
        dist = np.linalg.norm(mu[i:i+1,0:4]-mu[j:j+1,0:4])
        if dist>d:
            d = dist

# Implement RBF:
w = np.zeros((K,1))         # Kx1
hidden = np.zeros((K,1))            # Kx1
P = 0.001 * np.identity(K)          #KxK
alpha = np.zeros((1,1))         #1x1

for i in range(size):
    for j in range(K):
        hidden[j,0] = np.exp((np.linalg.norm(training_set[i:i+1,0:4]-mu[j:j+1,0:4])**2)/(-1 * d**2/K))
    P = P - np.matmul(np.matmul(P,hidden),np.matmul(np.transpose(hidden),P))/(1+np.matmul(np.matmul(np.transpose(hidden),P),hidden))
    g = np.matmul(P,hidden)

    alpha = training_set[i:i+1,4:5] - np.matmul(np.transpose(w),hidden)
    w = w + np.matmul(g,alpha)

    # Error on test set
    e = 0
    for j in range(100-size):
        for k in range(K):
            hidden[k,0] = np.exp((np.linalg.norm(test_set[j:j+1,0:4]-mu[k:k+1,0:4])**2)/(-1 * d**2/K))
        expect = np.matmul(np.transpose(w),hidden)
        
        if expect[0,0] * test_set[j,4] <= 0:
            e = e + 1
    x.append(i)
    
    error1.append(e/30)

plt.plot(x,error1)
plt.show()



    



