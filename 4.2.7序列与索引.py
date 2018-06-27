# -*- coding: utf-8 -*-
"""
Created on Tue May 22 14:16:24 2018

@author: lWX379138
"""
import tensorflow as tf
import numpy as np

with tf.Session() as sess:
    t = [[1,2,3],[3,4,5],[-1,3,2]]
    i = tf.argmin(t,0)      #返回最小值的索引，0 表示按列， 1 表示按行
    print(sess.run(i))
    i = tf.argmin(t,1)
    print(sess.run(i))
    
    i = tf.argmax(t,1)
    print(sess.run(i))    
    
    t1 = [11,22,4]
    t2 = [5,3,4]
    dif = tf.setdiff1d(t1,t2)
    print(sess.run(dif))    
    
    cond = [True,False,False,True]
    x = [1,2,3,4]
    y = [5,6,7,8]
    print(sess.run(tf.where(cond)))
    print(sess.run(tf.where(cond,x,y)))

    x = [1,1,2,4,4,4,7,8,8,4]
    y,idx = tf.unique(x) #y为x列表中唯一化数据列表，idx为x数据对应y元素的index
    print("y=",sess.run(y))   
    print("idx=",sess.run(idx))
    
    
    x= [3,4,0,2,1,5]
    t = tf.invert_permutation(x)
    print(sess.run(t))
    
    #x = [[1,2],[4,9]]
    #t = tf.random_shuffle(x)
    #print(sess.run(x))