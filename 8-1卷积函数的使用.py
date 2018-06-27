# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 14:32:23 2018

@author: lWX379138
"""
import tensorflow as tf
import numpy as np

#定义输入变量
#[batch in_weight,in_height,in_channels] 
#[训练时一个批次的数量，图片的高度，图片的宽度，图片的通道数]
input = tf.Variable(tf.constant(1.0,shape = [ 1,5,5,1]))    # 1 通道
input2 =tf.Variable(tf.constant(1.0,shape = [ 1,5,5,2]))    # 2 通道
input3 = tf.Variable(tf.constant(1.0,shape= [ 1,4,4,1]))    # 1 通道

#定义卷积变量
#[filter_height,filter_weight,in_channels,outchannels]
#[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]
filter1 = tf.Variable(tf.constant([-1.0,0,0,-1.0],shape=[2,2,1,1]))                 #1 通道
filter2 = tf.Variable(tf.constant([-1.0,0,0,-1.0,-1.0,0,0,-1.0],shape=[2,2,1,2]))   #1 通道
filter3 = tf.Variable(tf.constant([-1.0,0,0,-1,-1.0,0,0,-1,-1.0,0,0,-1],shape=[2,2,1,3]))   #1 通道
filter4 = tf.Variable(tf.constant([-1.0,0,0,-1,
                                   -1.0,0,0,-1,
                                   -1.0,0,0,-1,
                                   -1.0,0,0,-1],shape = [2,2,2,2]))             #2 通道
filter5 = tf.Variable(tf.constant([-1.0,0,0,-1,-1.0,0,0,-1],shape=[2,2,2,1]))   #2 通道

#定义卷积操作
#padding 的值'VALID',表示边缘不填充， ’SAME‘，表示填充到卷积核可以到达边缘
op1 = tf.nn.conv2d(input,filter1,strides=[1,2,2,1],padding='SAME')     #1个通道输入，生成1个feature map
op2 = tf.nn.conv2d(input,filter2,strides=[1,2,2,1],padding='SAME')     #1个通道输入，生成2个feature map
op3 = tf.nn.conv2d(input,filter3,strides=[1,2,2,1],padding='SAME')     #1个通道输入，生成3个feature map
op4 = tf.nn.conv2d(input2,filter4,strides=[1,2,2,1],padding='SAME')     #2个通道输入，生成4个feature map
op5 = tf.nn.conv2d(input2,filter5,strides=[1,2,2,1],padding='SAME')     #2个通道输入，生成5个feature map
op6 = tf.nn.conv2d(input3,filter1,strides=[1,2,2,1],padding='SAME')     #1个通道输入，生成2个feature map

vop1 = tf.nn.conv2d(input,filter1,strides=[1,2,2,1],padding='VALID')     #1个通道输入，生成2个feature map
vop6 = tf.nn.conv2d(input3,filter1,strides=[1,2,2,1],padding='VALID')     #1个通道输入，生成2个feature map 


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    
    print("op1:\n",sess.run([op1,filter1]))
    print("-----------------------------------\n")
    
    print("op2:\n",sess.run([op2,filter2]))
    print("op3:\n",sess.run([op3,filter2]))
    print("-----------------------------------\n")
    
    
    print("op4:\n",sess.run([op4,filter4]))
    print("op4:\n",sess.run([op5,filter5]))
    print("-----------------------------------\n")
    
    
    print("op1:\n",sess.run([op1,filter1]))
    print("vop1:\n",sess.run([vop1,filter1]))
    print("op6:\n",sess.run([op6,filter1]))
    print("vop6:\n",sess.run([vop6,filter1]))
    print("-----------------------------------\n")