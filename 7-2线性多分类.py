# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 16:06:12 2018

@author: lWX379138
"""
import tensorflow as tf
import numpy as np
from matplotlib.colors import *
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

def generate(sample_size,num_classes,mean,cov,diff,regression):
    samples_per_class=int(sample_size/num_classes)
    X0=np.random.multivariate_normal(mean,cov,samples_per_class)
    Y0=np.zeros(samples_per_class)
    for ci,d in enumerate(diff):
        X1=np.random.multivariate_normal(mean+d,cov,samples_per_class)
        Y1=(ci+1)*np.ones(samples_per_class)
        X0=np.concatenate((X0,X1))
        Y0=np.concatenate((Y0,Y1))
    
    print(np.shape(Y0))
    
    Y=np.zeros((samples_per_class*num_classes,3))
    print(np.shape(Y))
    if regression == False: #one-hot编码，将 0 转换成 1 0 
        #class_ind = [ Y == class_number for class_number in range(num_classes)]
        #Y = np.asarray(np.hstack(class_ind),dtype=np.float32)
        
        Y[Y0==0,0]=1
        Y[Y0==1,1]=1
        Y[Y0==2,2]=1
        pass
    #print(len(X0),len(Y0))
    X, Y = shuffle(X0,Y)
    return X,Y
    #np.random.shuffle(X0)
    #np.random.shuffle(Y0)
    #return X0,Y0
pass

input_dim = 2 
np.random.seed(10)
num_classes=3
mean=np.random.randn(input_dim )
cov=np.eye(input_dim )
X, Y=generate(1000,num_classes,mean,cov,[[3.0,3.0],[3.0,0]],False)
aa=[np.argmax(i) for i in Y]
colors=['r' if l==0 else 'b' if l ==1 else 'y' for l in aa[:]]
plt.scatter(X[:,0],X[:,1],c=colors)
plt.xlabel("Scaled age (in yrs)")
plt.ylabel("Tumor size (in cm)")
plt.show()

lab_dim = num_classes
#定位符
input_features = tf.placeholder(tf.float32,[None,input_dim])
input_lables = tf.placeholder(tf.float32,[None,lab_dim])
#定义学习参数
W = tf.Variable(tf.random_normal([input_dim,lab_dim]),name = 'weight')
b = tf.Variable(tf.zeros([lab_dim]),name='bias')
output = tf.matmul(input_features,W) + b

z = tf.nn.softmax( output )

a1 = tf.argmax(tf.nn.softmax( output ) ,axis = 1)   #按行找出最大索引，生成数组
b1 = tf.argmax(input_lables,axis = 1)
err = tf.count_nonzero( a1 -b1 )                    #两个数组相减，不为 0 的 就是错误个数

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = input_lables ,logits = output)
loss = tf.reduce_mean(cross_entropy)                #对交叉熵取均值很有必要

optimizer = tf.train.AdamOptimizer(0.04)            #尽量用 Adam 算法的优化器函数，因其收敛快，会动态调节梯度
train = optimizer.minimize(loss)

maxEpochs = 50
minibatchSize = 25

#启动session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(maxEpochs):
        sumerr = 0 
        for i in range(np.int32(len(Y)/minibatchSize)):
            x1 = X[i*minibatchSize:(i+1)*minibatchSize]
            y1 = Y[i*minibatchSize:(i+1)*minibatchSize]
            
            _,lossval,outputval,errval = sess.run([train,loss,output,err],feed_dict = {input_features:x1 , input_lables:y1})
            sumerr = sumerr + (errval/minibatchSize)
            
            pass
        print("Epoch:","%04d"%(epoch+1),"cost=","%.9f"%(lossval),"err=",sumerr/minibatchSize)
        pass
    
    input_dim = 2 
    np.random.seed(10)
    num_classes=3
    mean=np.random.randn(input_dim )
    cov=np.eye(input_dim )
    X, Y=generate(200,num_classes,mean,cov,[[3.0,3.0],[3.0,0]],False)
    aa=[np.argmax(i) for i in Y]
    colors=['r' if l==0 else 'b' if l ==1 else 'y' for l in aa[:]]
    plt.scatter(X[:,0],X[:,1],c=colors)
    plt.xlabel("Scaled age (in yrs)")
    plt.ylabel("Tumor size (in cm)")
    
    #'''
    x = np.linspace(-1,8,200)
    
    y = -x*(sess.run(W)[0][0]/sess.run(W)[1][0] ) - sess.run(b)[0]/sess.run(W)[1][0]
    plt.plot(x,y,label = "first line",lw = 3)
    
    y = -x*(sess.run(W)[0][1]/sess.run(W)[1][1] ) - sess.run(b)[1]/sess.run(W)[1][1]
    plt.plot(x,y,label = "second line",lw = 2)
    
    y = -x*(sess.run(W)[0][2]/sess.run(W)[1][2] ) - sess.run(b)[2]/sess.run(W)[1][2]
    plt.plot(x,y,label = "third line",lw = 1)
    
    #'''
    plt.legend()
    plt.show()
    print(sess.run(W),sess.run(b))


    input_dim = 2 
    np.random.seed(10)
    num_classes=3
    mean=np.random.randn(input_dim )
    cov=np.eye(input_dim )
    X, Y=generate(200,num_classes,mean,cov,[[3.0,3.0],[3.0,0]],False)
    aa=[np.argmax(i) for i in Y]
    colors=['r' if l==0 else 'b' if l ==1 else 'y' for l in aa[:]]
    plt.scatter(X[:,0],X[:,1],c=colors)
    plt.xlabel("Scaled age (in yrs)")
    plt.ylabel("Tumor size (in cm)")
    
    nb_of_xs = 200    
    xs1 = np.linspace(-1,8,num=nb_of_xs)
    xs2 = np.linspace(-1,8,num=nb_of_xs)
    xx, yy = np.meshgrid(xs1,xs2)   #创建网格
    #初始化和填充
    classification_plane = np.zeros((nb_of_xs,nb_of_xs))
    for i in range(nb_of_xs):
        for j in range(nb_of_xs):
            classification_plane[i,j] = sess.run(a1,feed_dict={input_features:[[xx[i,j],yy[i,j]]]})
            pass
        pass
    #创建color map 显示
    cmap = ListedColormap([
            colorConverter.to_rgba('r',alpha=0.30),
            colorConverter.to_rgba('b',alpha=0.30),
            colorConverter.to_rgba('y',alpha=0.30),
            ])
    #显示各个样本边界
    plt.contour(xx,yy,classification_plane,cmap=cmap)
    plt.show()

















