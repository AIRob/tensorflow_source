# -*- coding: utf-8 -*-
"""
Created on Tue May 22 11:31:04 2018

@author: lWX379138
"""
import tensorflow as tf
import numpy as np

with tf.Session() as sess:
    x = [[1,1,1],[1,1,1]]
    print(sess.run(tf.reduce_sum(x)))
    print(sess.run(tf.reduce_sum(x,0)))
    print(sess.run(tf.reduce_sum(x,1)))
    print(sess.run(tf.reduce_sum(x,1,keep_dims = True)))
    print(sess.run(tf.reduce_sum(x,[0,1])))

    pass
pass

print("reduce_prod 0")

fi = tf.Variable(tf.constant([2,3,4,5],shape=[2,2]))
ff = tf.reduce_prod(fi,0)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(fi))
    print(sess.run(ff))
    
print("reduce_prod 1")

fi = tf.Variable(tf.constant([2,3,4,5],shape=[2,2]))
ff = tf.reduce_prod(fi,1)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(fi))
    print(sess.run(ff))
    print(sess.run(tf.reduce_max(fi)))
    print(sess.run(tf.reduce_min(fi)))
    print(sess.run(tf.reduce_mean(fi)))
    
    #逻辑 '与'
    x = [[True,True],[False,False]]
    rx = tf.reduce_all(x)
    rx0 = tf.reduce_all(x,0)
    rx1 = tf.reduce_all(x,1)
    print("rx=",sess.run(rx)," rx0= ",sess.run(rx0)," rx1=",sess.run(rx1))
    #逻辑 '或'
    x = [[True,True],[False,False]]
    rx = tf.reduce_any(x)
    rx0 = tf.reduce_any(x,0)
    rx1 = tf.reduce_any(x,1)
    print("rx=",sess.run(rx)," rx0= ",sess.run(rx0)," rx1=",sess.run(rx1))
    