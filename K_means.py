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
data_class_setosa = np.concatenate((data_setosa,np.ones((50,1))),axis = 1)
data_class_versicolor = np.concatenate((data_versicolor,np.ones((50,1))*2),axis = 1)
data_class_virginica = np.concatenate((data_virginica,np.ones((50,1))*3),axis = 1)
data_class = np.concatenate((data_class_setosa,data_class_versicolor,data_class_virginica),axis = 0)
np.random.shuffle(data_class)
percentage = 0.7                # Percentage of training set
size = 105
training_set = data_class[0:int(percentage*150),:]       # Training data
test_set = data_class[int(percentage*150):150,:]         # Test data

# We implement K means clustering

K  = 1

while K < 25:
    mu = np.random.rand(K,4)          # Kx4 mean vector
    for t in range(K):
        mu[t,0] = mu[t,0] + 6
        mu[t,1] = mu[t,1] + 2.5
        mu[t,2] = mu[t,2] + 4.2
        mu[t,3] = mu[t,3] + 1.5
    C = np.zeros((size,1))
    C1 = np.zeros((150-size,1))
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
    
    '''
    # Calculate error for test set
    error = 0
    for j in range(150-size):
        dist = np.zeros((K,1))
        for k in range(K):
            dist[k,0] = np.linalg.norm(test_set[j:j+1,0:4]-mu[k:k+1,0:4])
        error = error + np.amin(dist)
    error1.append(error/(150-size))
    '''
    
    # Update coder for test
    lc = np.zeros((K,3))
    for j in range(150-size):
        dist = np.zeros((K,1))
        for k in range(K):
            dist[k,0] = np.linalg.norm(test_set[j:j+1,0:4]-mu[k:k+1,0:4])
        k_close = np.argmin(dist)
        C[j,0] = int(k_close)
        lc[int(k_close),int(test_set[j,4])-1] = lc[int(k_close),int(test_set[j,4])-1] + 1
    
    k_max = np.argmax(lc,axis = 1)
    
    error = 0
    for k in range(K):
        t = k_max[k]
        if lc[k,0]==0 and lc[k,1]==0 and lc[k,2]==0:
            error=0
        else:
            error = error + 1 - lc[k,t]/(lc[k,0]+lc[k,1]+lc[k,2])
        
    error = error/K
    error1.append(error)

        
    x.append(K)
    K = K + 1
    print(K)

plt.plot(x,error1)
plt.show()
    



