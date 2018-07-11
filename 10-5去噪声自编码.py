# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 15:09:47 2018

@author: lWX379138
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#导入MINST 数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/",one_hot=True)

train_X = mnist.train.images
train_Y = mnist.train.labels
test_X  = mnist.test.images
test_Y  = mnist.test.labels

n_input = 784
n_hidden_1 = 256

#占位符
x = tf.placeholder("float",[None,n_input])
y = tf.placeholder("float",[None,n_input])
dropout_keep_prob = tf.placeholder("float")

#学习参数
weights = {"h1":tf.Variable(tf.random_normal([n_input,n_hidden_1])),
           "h2":tf.Variable(tf.random_normal([n_hidden_1,n_hidden_1])),
           "out":tf.Variable(tf.random_normal([n_hidden_1,n_input]))
           }
biases = {"b1":tf.Variable(tf.zeros([n_hidden_1])),
          "b2":tf.Variable(tf.zeros([n_hidden_1])),
          'out':tf.Variable(tf.zeros([n_input]))
          }

#网络模型
def denoise_auto_encoder(_X,_weights,_biases,_keep_prob):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(_X,_weights['h1']),_biases['b1']))
    layer_1out = tf.nn.dropout(layer_1,_keep_prob)
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1out,_weights['h2']),_biases['b2']))
    layer_2out = tf.nn.dropout(layer_2,_keep_prob)
    return tf.nn.sigmoid(tf.matmul(layer_2out,_weights['out']) + _biases['out'])
pass

reconstruction = denoise_auto_encoder(x,weights,biases,dropout_keep_prob)

#cost 计算
cost = tf.reduce_mean(tf.pow(reconstruction - y , 2) )
#优化器
optm = tf.train.AdamOptimizer(0.01).minimize(cost)

#训练参数
epochs = 20
batch_size = 256
disp_step = 2

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("开始训练")
    for epoch in range(epochs):
        num_batch = int(mnist.train.num_examples / batch_size)
        total_cost = 0.
        for i in range(num_batch):
            batch_xs , batch_ys = mnist.train.next_batch(batch_size)
            batch_xs_noisy = batch_xs + 0.3*np.random.randn(batch_size,784)     #引入噪声
            feeds = {x:batch_xs_noisy,y:batch_xs,dropout_keep_prob:0.5}
            sess.run(optm,feed_dict=feeds)
            total_cost += sess.run(cost,feed_dict=feeds)
            pass
        #显示训练信息
        if epoch % disp_step == 0:
            print("Epoch %02d/%02d average cost : %.6f"%(epoch,epochs,total_cost/num_batch))
            pass
        pass
    print("训练完成")
    #数据可视化
    show_num = 10
    test_noisy = mnist.test.images[:show_num] + 0.3*np.random.randn(show_num,784)
    encode_decode = sess.run(reconstruction,feed_dict={x:test_noisy,dropout_keep_prob:0.5})
    f,a = plt.subplots(3,10,figsize = (10,3))
    for i in range(show_num):
        a[0][i].imshow(np.reshape(test_noisy[i],(28,28)))                               #加入噪声后的图片
        a[1][i].imshow(np.reshape(mnist.test.images[i],(28,28)))                        #原始图片
        a[2][i].imshow(np.reshape(encode_decode[i],(28,28))    ,cmap=plt.get_cmap('gray'))  #第一张图片 去燥后的图片，灰度图
        pass
    plt.show()
    
    #测试鲁棒性
    randidx = np.random.randint(test_X.shape[0],size = 1)
    orgvec = test_X[randidx,:]
    testvec = test_X[randidx,:]
    label = np.argmax(test_Y[randidx,:],1)
    
    print("label is %d" % (label))
    #噪声类型
    print("Salt and Pepper Noise")
    noisyvec = testvec
    rate = 0.15
    noiseidx = np.random.randint(test_X.shape[1],size=int(test_X.shape[1]*rate))
    noisyvec[0,noiseidx] = 1-noisyvec[0,noiseidx]
    outvec = sess.run(reconstruction,feed_dict={x:noisyvec,dropout_keep_prob:0.5})
    outimg = np.reshape(outvec,(28,28))
    
    #可视化
    plt.matshow(np.reshape(orgvec,(28,28)) ,cmap=plt.get_cmap('gray'))      #原始图
    plt.title("Original Image")
    plt.colorbar()
    
    plt.matshow(np.reshape(noisyvec,(28,28)) ,cmap=plt.get_cmap('gray'))    #加入噪声的图
    plt.title("Input Image")
    plt.colorbar()
    
    plt.matshow(np.reshape(outimg,(28,28)) ,cmap=plt.get_cmap('gray'))      #去燥后的图
    plt.title("Original Image")
    plt.colorbar()    
    
    plt.show()
    
    