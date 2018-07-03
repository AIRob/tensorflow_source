# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 11:03:05 2018

@author: lWX379138
"""
import tensorflow as tf
from tensorflow.contrib import rnn
#输入MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('data/',one_hot=True)

#定义参数
learning_rate = 0.001 
training_iters = 100000
batch_size = 128
display_step = 10

#网络模型参数设置
n_input = 28                #MNIST 数据输入（img: 28 * 28)
n_steps = 28                #步骤序列
n_hidden = 128              #隐藏层个数
n_classes = 10              #MNIST 总类别 (0 ~ 9)

tf.reset_default_graph()

#定义占位符
x = tf.placeholder('float',[None,n_steps,n_input])
y = tf.placeholder('float',[None,n_classes])

x1 = tf.unstack(x,n_steps,1)
lstm_fw_cell = rnn.BasicLSTMCell(n_hidden,forget_bias = 1.0)
#反向cell
lstm_bw_cell = rnn.BasicLSTMCell(n_hidden,forget_bias = 1.0)
outputs,_ ,_ = rnn.static_bidirectional_rnn(lstm_fw_cell,lstm_bw_cell,x1,dtype = tf.float32)

print(outputs[0].shape,len(outputs))

pred = tf.contrib.layers.fully_connected(outputs[-1],n_classes,activation_fn = None)

#定义参数
learning_rate = 0.01
batch_size = 128
time_steps = 28
#损失函数 交叉熵 P107页，最后一行
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))#tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),reduction_indices = 1)) 
#等价于 
#cost = tf.nn.softmax_cross_entropy_with_logits(labels = y,logits=pred)

#使用梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
#测试model 校正预测
correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#'''
#启动session 1 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) #initializing OP
    
    iter=1
    while iter<800:
        batch_x,batch_y=mnist.train.next_batch(batch_size=batch_size)
        batch_x=batch_x.reshape((batch_size,time_steps,n_input))
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if iter %100==0:
            acc=sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
            los=sess.run(cost,feed_dict={x:batch_x,y:batch_y})
            print("For iter ",iter)
            print("Accuracy ",acc)
            print("Loss ",los)
            print("__________________")
        iter=iter+1
    #calculating test accuracy
    test_data = mnist.test.images[:128].reshape((-1, time_steps, n_input))
    test_label = mnist.test.labels[:128]
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label})) 