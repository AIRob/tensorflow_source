# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 10:16:28 2018

@author: lWX379138
"""
import tensorflow as tf
import numpy as np

'''
##max_pool_with_argmax只支持GPU操作
'''
def max_pool_with_argmax(net,stride):
    _,mask = tf.nn.max_pool_with_argmax(net,ksize = [1,stride,stride,1],strides=[1,stride,stride,1],padding='SAME') ##max_pool_with_argmax只支持GPU操作
    mask = tf.stop_gradient(mask)
    net = tf.nn.max_pool(net,ksize=[1,stride,stride,1],strides=[1,stride,stride,1],padding='SAME')
    return net,mask
pass

img = tf.constant([
        [[0.0,4.0],[0.0,4.0],[0.0,4.0],[0.0,4.0]],
        [[1.0,5.0],[1.0,5.0],[1.0,5.0],[1.0,5.0]],
        [[2.0,6.0],[2.0,6.0],[2.0,6.0],[2.0,6.0]],
        [[3.0,7.0],[3.0,7.0],[3.0,7.0],[3.0,7.0]]
    ])    

img = tf.reshape(img,[1,4,4,2])
pooling2 = tf.nn.max_pool(img,[1,2,2,1],[1,2,2,1],padding = 'SAME')
encode,mask = max_pool_with_argmax(img,2)    
with tf.Session() as sess:
    print("image:")
    image = sess.run(img)
    print(image)
    result = sess.run(pooling2)
    print("pooling2:\n",result)
    result , mask2 = sess.run(encode,mask)
    print("encode:\n",result,mask2)
    
    pass
pass