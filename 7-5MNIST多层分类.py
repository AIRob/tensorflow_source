# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 16:04:36 2018

@author: lWX379138
"""
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot = True)
import pylab

tf.reset_default_graph()

#定义参数
learning_rate = 0.001
training_epochs = 25
batch_size = 100
display_step = 1


#设置网络模型参数
n_hidden_1 = 256    #第一个隐藏层节点个数
n_hidden_2 = 256    #第二个隐藏层节点个数
n_input = 784       #MNIST共784（ 28x28）维
n_classes = 10      #MNIST共10个类别

#定义占位符
x = tf.placeholder(tf.float32,[None,n_input])
y = tf.placeholder(tf.float32,[None,n_classes])

#创建model

def multilayer_perceptron(x , weights , biases):
    #第一层隐藏层
    layer_1 = tf.add(tf.matmul(x , weights['h1']),biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    #第二层隐藏层
    layer_2 = tf.add(tf.matmul(layer_1,weights['h2']),biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    #输出层
    out_layer = tf.matmul(layer_2,weights['out'])+biases['out']
    
    return out_layer
pass

#学习参数
weights = {
    'h1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
    'h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
    'out':tf.Variable(tf.random_normal([n_hidden_2,n_classes]))                                     
    }
biases = {
    'b1':tf.Variable(tf.random_normal([n_hidden_1])),
    'b2':tf.Variable(tf.random_normal([n_hidden_2])),
    'out':tf.Variable(tf.random_normal([n_classes]))
    }

#输出值
#'''
pred = multilayer_perceptron(x , weights, biases)

#定义loss和优化器
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

'''

#启动session 1 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) #initializing OP
    
    #启动循环训练任务
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        #循环所有数据集
        for i in range(total_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            #运行优化器
            _, c = sess.run([optimizer,cost],feed_dict={x:batch_xs,y:batch_ys})
            #计算平均值loss
            avg_cost += c/total_batch
            pass
        #显示训练中的详细信息
        if (epoch +1 ) % display_step == 0:
            print("Epoch:","%04d"%(epoch+1),"cost=","{:.9f}".format(avg_cost))
            pass
        pass
    
    print("train Finished!")
    
    #测试model 校正预测
    correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    #计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    print("Accuracy:",accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))
        #保存模型
    saver = tf.train.Saver()
    model_path = "log/521model.ckpt"
    save_path = saver.save(sess,model_path)
    print("Model saved in file:%s"%save_path)
#'''

#'''
print("Starting 2nd session...")
with tf.Session() as sess:
    #初始化变量
    sess.run(tf.global_variables_initializer())
    #恢复模型变量
    #保存模型
    saver = tf.train.Saver()
    model_path = "log/521model.ckpt"
    saver.restore(sess,model_path)
    
    #测试model
    correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    #计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    print("Accuracy:",accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))
    
    output = tf.argmax(pred,1)
    batch_xs,batch_ys = mnist.train.next_batch(4)
    outputval,predv = sess.run([output,pred],feed_dict={x:batch_xs,y:batch_ys})
    print(outputval,predv)
    print(batch_xs.shape,batch_ys)
    
    im = batch_xs[0]
    im = im.reshape(-1,28)
    pylab.imshow(im)
    pylab.show()

    im = batch_xs[1]
    im = im.reshape(-1,28)
    pylab.imshow(im)
    pylab.show()
    
    im = batch_xs[2]
    im = im.reshape(-1,28)
    pylab.imshow(im)
    pylab.show()
    
    im = batch_xs[3]
    im = im.reshape(-1,28)
    pylab.imshow(im)
    pylab.show()

    print("End of 2nd session !")    
#'''   
