# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 10:45:20 2018

@author: lWX379138
"""
import tensorflow as tf

w1 = tf.Variable([[1,2]])
w2 = tf.Variable([[3,4]])
b = tf.Variable(2)

y = tf.matmul(w1 , [[9],[10]])+b
grads = tf.gradients(y,[w1])    #求w1的梯度

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    gradval = sess.run(grads)
    print(gradval)

