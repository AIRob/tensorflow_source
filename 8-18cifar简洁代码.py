# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 19:02:17 2018

@author: lWX379138
"""

import cifar10_input
import tensorflow as tf
import numpy as np


batch_size = 128 
data_dir = 'cifar-10-binary/cifar-10-batches-bin'
print('begin')
images_train,labels_train = cifar10_input.inputs(eval_data = False, data_dir = data_dir, batch_size = batch_size) 
images_test,labels_test   = cifar10_input.inputs(eval_data = True,  data_dir = data_dir, batch_size = batch_size) 

print('begin data')

def weight_variable(shape):
    '''
    初始化权重
    
    args:
        shape：权重shape
    '''
    initial = tf.truncated_normal(shape=shape,mean=0.0,stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    '''
    初始化偏置
    
    args:
        shape:偏置shape
    '''
    initial =tf.constant(0.1,shape=shape)
    return tf.Variable(initial)


def conv2d(x,W):
    '''
    卷积运算 ，使用SAME填充方式   池化层后
         out_height = in_hight / strides_height（向上取整）
         out_width = in_width / strides_width（向上取整）
    
    args:
        x:输入图像 形状为[batch,in_height,in_width,in_channels] 
        W:权重 形状为[filter_height,filter_width,in_channels,out_channels]        
    '''
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')


def max_pool_2x2(x):
    '''
    最大池化层,滤波器大小为2x2,'SAME'填充方式  池化层后
         out_height = in_hight / strides_height（向上取整）
         out_width = in_width / strides_width（向上取整）
    
    args:
        x:输入图像 形状为[batch,in_height,in_width,in_channels] 
    '''
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def avg_pool_6x6(x):
    '''
    全局平均池化层，使用一个与原有输入同样尺寸的filter进行池化，'SAME'填充方式  池化层后
         out_height = in_hight / strides_height（向上取整）
         out_width = in_width / strides_width（向上取整）
    
    args；
        x:输入图像 形状为[batch,in_height,in_width,in_channels] 
    '''
    return tf.nn.avg_pool(x,ksize=[1,6,6,1],strides=[1,6,6,1],padding='SAME')

#定义占位符
x = tf.placeholder(dtype=tf.float32,shape=[None,24,24,3])  #cifar data 的 shape 24x24x3

y = tf.placeholder(dtype=tf.float32,shape=[None,10]) #0~9 数字分类 =》 10 classes

#W_conv1 = weight_variable([5,5,3,64])
#b_conv1 = bias_variable([64])

x_image = tf.reshape(x,[-1,24,24,3])

#h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1 )
#h_pool1 = max_pool_2x2(h_conv1)
h_conv1 = tf.contrib.layers.conv2d(x_image,64,5,1,'SAME',activation_fn = tf.nn.relu)
h_pool1 = tf.contrib.layers.max_pool2d(h_conv1,[2,2],stride = 2 ,padding='SAME')


#W_conv2 = weight_variable([5,5,64,64])
#b_conv2 = bias_variable([64])

#h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2 )    # h_pool1
#h_pool2 = max_pool_2x2(h_conv2)
h_conv2 = tf.contrib.layers.conv2d(h_pool1,64,[5,5],1,'SAME',activation_fn = tf.nn.relu)
h_pool2 = tf.contrib.layers.max_pool2d(h_conv2,[2,2],stride = 2 ,padding='SAME')


#W_conv3 = weight_variable([5,5,64,10])
#b_conv3 = bias_variable([10])
#h_conv3 = tf.nn.relu(conv2d(h_pool2,W_conv3) + b_conv3)
 
#nt_hpool3 = avg_pool_6x6(h_conv3)   #10
#nt_hpool3_flat = tf.reshape(nt_hpool3,[-1,10])
nt_hpool2 = tf.contrib.layers.avg_pool2d(h_pool2,[6,6],stride=6,padding='SAME')
nt_hpool2_flat = tf.reshape(nt_hpool2,[-1,64])

#y_conv = tf.nn.softmax(nt_hpool3_flat)
y_conv = tf.contrib.layers.fully_connected(nt_hpool2_flat,10,activation_fn = tf.nn.softmax)

cross_entropy = -tf.reduce_sum(y*tf.log(y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners(sess = sess)
for i in range(15000):#20000
    image_batch ,label_batch = sess.run([images_train,labels_train])
    label_b = np.eye(10,dtype=float)[label_batch]      #onehot 编码
    
    train_step.run(feed_dict = {x:image_batch,y:label_b},session = sess)
    #'''
    if i % 200 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:image_batch,y:label_b},session = sess )
        print("step:%d,training accuracy %g"%(i,train_accuracy))
        pass
    #'''
    pass
pass
#'''
image_batch ,label_batch = sess.run([images_test,labels_test])
label_b = np.eye(10,dtype=float)[label_batch]
print("Finished\ntest accuracy:%g"%accuracy.eval(feed_dict={x:image_batch,y:label_b},session = sess))
#'''
#保存模型
saver = tf.train.Saver()
model_path = "log/8-18model.ckpt"
save_path = saver.save(sess,model_path)
print("Model saved in file:%s"%save_path)
    
    
print('end')





