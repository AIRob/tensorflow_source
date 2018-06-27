# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 17:08:20 2018

@author: lWX379138
"""
import tensorflow as tf
import numpy as np

#模拟数据
img = tf.Variable(tf.constant(1.0,shape= [ 1,4,4,1]))

filter = tf.Variable(tf.constant([1.0,0,-1,-2],shape = [2,2,1,1]))
#分别进行SAME和VALID操作
conv = tf.nn.conv2d(img,filter,strides=[1,2,2,1],padding='VALID')
cons = tf.nn.conv2d(img,filter,strides=[1,2,2,1],padding='SAME')
print(conv.shape)
print(cons.shape)

#再进行反卷积
contv = tf.nn.conv2d_transpose(conv , filter,[1,4,4,1],strides = [1,2,2,1],padding = 'VALID')
conts = tf.nn.conv2d_transpose(cons,  filter,[1,4,4,1],strides = [1,2,2,1],padding = 'SAME')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    print('conv:\n',sess.run([conv,filter]))
    print('cons:\n',sess.run([cons]))
    print('contv:\n',sess.run([contv]))
    print('conts:\n',sess.run([conts]))



