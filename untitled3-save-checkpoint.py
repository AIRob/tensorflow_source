# -*- coding: utf-8 -*-
"""
Created on Mon May 21 09:14:55 2018

@author: lWX379138
为模型添加保存检查点
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#定义生成loss可视化的函数

plotdata = {"batchsize":[],"loss":[]}

def moving_average(a,w=10):
    if len(a) < w:
        return a[:]
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx,val in enumerate(a)]
    pass
pass

#生成模拟数据
train_X = np.linspace(-1,1,100)
train_Y = 2*train_X +np.random.randn(*train_X.shape)*0.3  #y=2x,但是加入噪声

#图形显示
plt.plot(train_X,train_Y,label="Original data")
plt.legend()
plt.show()    

tf.reset_default_graph()

#创造模型
#占位符
X = tf.placeholder("float")
Y = tf.placeholder("float")    
#模型参数
W = tf.Variable(tf.random_normal([1]),name="weight")
b = tf.Variable(tf.zeros([1]),name="bias")
#前向结构
z = tf.multiply(X,W)+b

#反向优化
cost = tf.reduce_mean(tf.square(Y-z))
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) #梯度下降

#初始化所有变量
init =  tf.global_variables_initializer()
#定义学习参数
training_epochs = 20
display_step = 2
saver = tf.train.Saver(max_to_keep = 20) #生成saver,max_to_keep为最大保存数
savedir = 'log/'

#'''
#启动图
with tf.Session() as sess:
    sess.run(init)
    
    #向模型中输入数据
    for epoch in range(training_epochs):
        for(x,y) in zip(train_X,train_Y):
            sess.run(optimizer,feed_dict={X:x,Y:y})
            pass
        #显示训练中的详细信息
        if epoch % display_step == 0:
            loss = sess.run(cost,feed_dict={X:train_X,Y:train_Y})
            pass
        print("Epoch:",epoch+1,"cost=",loss,"w=",sess.run(W),"b=",sess.run(b))
        if not (loss == "NA"):
            plotdata['batchsize'].append(epoch)
            plotdata['loss'].append(loss)
            pass
        print("epoch=",epoch)
        saver.save(sess,savedir+"linermodel.cpkt",global_step = epoch)#,global_step = epoch)
        pass
    print("Finished!")
    
    print("cost=",sess.run(cost,feed_dict={X:train_X,Y:train_Y}),"W=",sess.run(W),"b=",sess.run(b))
    
    #显示模型
    plt.plot(train_X,train_Y,'ro',label="Origial data")
    plt.plot(train_X,sess.run(W)*train_X+sess.run(b),label="Fitted Wlines")
    plt.legend()
    plt.show()
    
    plotdata["avgloss"] = moving_average(plotdata["loss"])
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata["batchsize"],plotdata["avgloss"],"b--")
    plt.xlabel("Minibatch number")
    plt.ylabel("Losss")
    plt.title("Minibatch run vs . Training loss")
    
    plt.show()
    
    pass
pass
#'''
#重启一个sess ，载入检查点
load_epoch = 18
with tf.Session() as sess2:
    sess2.run(tf.global_variables_initializer())
    saver.restore(sess2,savedir + "linermodel.cpkt-"+str(load_epoch))
    print("X= 0.2,s=",sess2.run(z,feed_dict={X:0.2}))    
#'''    
    
    




        
        






