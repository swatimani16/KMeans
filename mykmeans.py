# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 19:50:59 2019

@author: Swati
"""
import copy
import numpy as np
import matplotlib.pyplot as plt
#Finding the 2-D random Guassian Data and converging the two datasets
dataset1=np.random.multivariate_normal([1,0],[[0.9,0.4],[0.4,0.9]],500)
dataset2=np.random.multivariate_normal([0,1.5],[[0.9,0.4],[0.4,0.9]],500)
X=np.append(dataset1,dataset2,axis=0)

print(X)
m=X.shape[0] #number of training examples
n=X.shape[1] #number of features. 
n_iter=10000

#input the clusters
K=int(input('Enter the number of clusters: '))

#finding the centers
c=[]
for i in range(K):
    x,y=input('Enter coordinates').split()
    c.append([x,y])


for i in range(len(c)):
    for j in range(len(c[0])):
        c[i][j]=float(c[i][j])
print(c)
c = np.array(c)

#finding transpose of the centers
c=c.T
print("centers:",c)
count=0

def mykmeans(X,K,c):
    #to store the output
    global count
    Output={}
    EuclidianDistance=np.array([]).reshape(m,0)
    for i in range(n_iter):
        #print("IN2")
        for k in range(K):
              tempDist=np.sqrt(np.sum((X-c[:,k])**2,axis=1))
              EuclidianDistance=np.c_[EuclidianDistance,tempDist]
        C=np.argmin(EuclidianDistance,axis=1)+1
        c_old=np.zeros(c.shape)
        c_old=copy.deepcopy(c)
        
        Y={}
        for k in range(K):
              Y[k+1]=np.array([]).reshape(2,0)
        for j in range(m):
              Y[C[j]]=np.c_[Y[C[j]],X[j]]
          #print("3rd",Y)
         
         
        for k in range(K):
              Y[k+1]=Y[k+1].T
        
        for k in range(K):
              #print (k)
              c[:,k]=np.mean(Y[k+1],axis=0)
        count=count+1
        if (np.linalg.norm((c_old -c), axis=None))<=0.001:
            break
       
        EuclidianDistance=np.array([]).reshape(m,0)
        Output=Y
    plt.scatter(X[:,0],X[:,1],c='black',label='unclustered data')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.title('Plot of data points')
    plt.show()
    color=['magenta','cyan','green','blue','red']
    labels=['cluster1','cluster2','cluster3','cluster4','cluster5']
    for k in range(K):
        plt.scatter(Output[k+1][:,0],Output[k+1][:,1],c=color[k],label=labels[k])
    plt.scatter(c[0,:],c[1,:],s=300,c='yellow',label='Centres')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.show()  
    return Output

clusters=mykmeans(X,K,c)
print('This is count',count)
#print(clusters)
