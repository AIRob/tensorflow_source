# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 10:14:12 2018

@author: lWX379138
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

def generate(sample_size,mean,cov,diff,regression):
    num_classes = 2
    samples_per_class = int(sample_size/2)
    
    X0 = np.random.multivariate_normal(mean,cov,samples_per_class)
    Y0 = np.zeros(samples_per_class)
    
    for ci,d in enumerate(diff):
        X1 = np.random.multivariate_normal(mean+d,cov,samples_per_class)
        Y1 = (ci+1)*np.ones(samples_per_class)
        
        X0 = np.concatenate((X0,X1))
        Y0 = np.concatenate((Y0,Y1))
        
        pass
    if regression == False: #one-hot编码，将 0 转换成 1 0 
        class_ind = [ Y == class_number for class_number in range(num_classes)]
        Y = np.asarray(np.hstack(class_ind),dtype=np.float32)
        pass
    #print(len(X0),len(Y0))
    X, Y = shuffle(X0,Y0)
    return X,Y
    #np.random.shuffle(X0)
    #np.random.shuffle(Y0)
    #return X0,Y0
pass

np.random.seed(10)
num_classes = 2
mean = np.random.randn(num_classes)
cov = np.eye(num_classes)
print(mean,cov)
X,Y = generate(1000,mean,cov,[3.0],True)
print(len(X),len(Y))
colors = ['r' if l == 0 else 'b' for l in Y[:]]
plt.scatter(X[:,0],X[:,1],c=colors)
plt.xlabel("Scaled age (in yrs)")
plt.ylabel("Tumor size (in cm)")
plt.show()
lab_dim = 1  
input_dim = 2

input_features = tf.placeholder(tf.float32,[None,input_dim])
input_lables = tf.placeholder(tf.float32,[None,lab_dim])
#定义学习参数
W = tf.Variable(tf.random_normal([input_dim,lab_dim],name="wight"))
b= tf.Variable(tf.zeros([lab_dim]),name="bias")

output = tf.nn.sigmoid(tf.matmul(input_features,W) + b)
cross_entropy = -(input_lables*tf.log(output) + (1 - input_lables)*tf.log(1 - output))
ser = tf.square(input_lables-output,name=None)
loss = tf.reduce_mean(cross_entropy)
err = tf.reduce_mean(ser)
optimizer = tf.train.AdamOptimizer(0.04)
#尽量使用这个，因其收敛快，会动态调节梯度
train = optimizer.minimize(loss) 

maxEpochs = 50
minibatchSize = 25

#启动session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    #向模型输入数据
    for epoch in range(maxEpochs):
        sumerr = 0 
        for i in range(np.int32(len(Y)/minibatchSize)):
            x1 = X[i*minibatchSize:(i+1)*minibatchSize,:]
            y1 = np.reshape(Y[i*minibatchSize:(i+1)*minibatchSize],[-1,1])
            tf.reshape(y1,[-1,1])
            _,lossval,outputval,errval = sess.run([train,loss,output,err],feed_dict={input_features:x1,input_lables:y1})
            sumerr = sumerr + errval
            pass
        print("Epoch:",'%04d'%(epoch+1),"cost=","{:.9f}".format(lossval),"err=",sumerr/minibatchSize)
        pass
    train_X ,train_Y = generate(100,mean,cov,[3.0],True)
    corlors = ['r' if l == 0 else  'b' for l in train_Y[:]]
    plt.scatter(train_X[:,0],train_X[:,1],c=colors)
    x = np.linspace(-1,8,200)
    print(sess.run(W)[0]/sess.run(W)[1],sess.run(b)/sess.run(W)[1])
    y = -x*(sess.run(W)[0]/sess.run(W)[1])-sess.run(b)/sess.run(W)[1]
    plt.plot(x,y,label = "Fitted line")
    plt.legend()
    plt.show()

















 
    