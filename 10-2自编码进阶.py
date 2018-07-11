# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 10:33:51 2018

@author: lWX379138
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#导入MINST 数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/",one_hot=True)

#学习率
learning_rate = 0.01
n_hidden_1 = 256        #第一层256个节点
n_hidden_2 = 64         #第二层64个节点
n_hidden_3 = 16         #第三层 16
n_hidden_4 = 2          #第三层 2
n_input  = 784          #MNIST数据集图片的维度（28 * 28）

#占位符
x = tf.placeholder("float",[None,n_input])      #输入
y = x                                           #输出

#学习参数
weights = {'encoder_h1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
           'encoder_h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
           'encoder_h3':tf.Variable(tf.random_normal([n_hidden_2,n_hidden_3])),
           'encoder_h4':tf.Variable(tf.random_normal([n_hidden_3,n_hidden_4])),
           
           "decoder_h1":tf.Variable(tf.random_normal([n_hidden_4,n_hidden_3])),
           'decoder_h2':tf.Variable(tf.random_normal([n_hidden_3,n_hidden_2])),
           "decoder_h3":tf.Variable(tf.random_normal([n_hidden_2,n_hidden_1])),
           'decoder_h4':tf.Variable(tf.random_normal([n_hidden_1,n_input])),           
           }
biases = {'encoder_b1':tf.Variable(tf.zeros([n_hidden_1])),
          'encoder_b2':tf.Variable(tf.zeros([n_hidden_2])),
          'encoder_b3':tf.Variable(tf.zeros([n_hidden_3])),
          'encoder_b4':tf.Variable(tf.zeros([n_hidden_4])),
          
          'decoder_b1':tf.Variable(tf.zeros([n_hidden_3])),
          'decoder_b2':tf.Variable(tf.zeros([n_hidden_2])),          
          'decoder_b3':tf.Variable(tf.zeros([n_hidden_1])),
          'decoder_b4':tf.Variable(tf.zeros([n_input]))
          }

#编码
def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x,weights['encoder_h1']),biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weights['encoder_h2']),biases['encoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2,weights['encoder_h3']),biases['encoder_b3']))
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3,weights['encoder_h4']),biases['encoder_b4']))    
    return layer_4
pass

#解码
def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x,weights['decoder_h1']),biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weights['decoder_h2']),biases['decoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2,weights['decoder_h3']),biases['decoder_b3']))
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3,weights['decoder_h4']),biases['decoder_b4']))    
    return layer_4
pass

#输出节点
encoder_out = encoder(x)
y_pred = decoder(encoder_out)

#cost为y与pred的平方差
cost = tf.reduce_mean(tf.pow(y - y_pred , 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

#训练参数
training_epochs  = 20       #一共迭代20次
batch_size = 256            #每次取256个样本
display_step = 1            #迭代5次输出异常信息

#启动回话
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    total_batch = int(mnist.train.num_examples/batch_size)
    #开始训练
    for epoch in range(training_epochs):    #迭代
        for i in range(total_batch):
            batch_xs , batch_ys = mnist.train.next_batch(batch_size)    #取数据
            _ , c = sess.run([optimizer,cost] ,feed_dict={x:batch_xs})
            #训练模型
        if epoch % display_step == 0:       #显示日志信息
            print("Epoch :",'%04d'%(epoch + 1),"cost=","{:.9f}".format(c))
            pass

        pass
    print('完成')

    correct_prediction = tf.equal(tf.argmax(y_pred,1),tf.argmax(y,1))
    #计算错误率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))
    print("Accuracy:",1-accuracy.eval({x:mnist.test.images,y:mnist.test.images}))

    #可视化结果
    show_num = 10
    reconstruction = sess.run(y_pred,feed_dict={x:mnist.test.images[:show_num]})
    f , a = plt.subplots(2,10,figsize=(10,2))
    for i in range(show_num):
        a[0][i].imshow(np.reshape(mnist.test.images[i],(28,28)))
        a[1][i].imshow(np.reshape(reconstruction[i],(28,28)))
        pass
    plt.show()
    
    aa = [np.argmax(l) for l in mnist.test.labels]  #将onehot转成一般编码
    encoder_result = sess.run(encoder_out,feed_dict={x:mnist.test.images})
    plt.scatter(encoder_result[:,0],encoder_result[:,1],c=aa)
    plt.colorbar()
    plt.show()
    pass
pass

