# -*- coding: utf-8 -*-
"""
Created on Tue May 22 09:53:56 2018

@author: lWX379138
"""
import tensorflow as tf
import numpy as np



diagonal = [1,2,3,4]

with tf.Session() as sess:
    print(sess.run(tf.diag(diagonal))) #返回一个给定对角值的tensor 
    '''
    [[1 0 0 0]
     [0 2 0 0]
     [0 0 3 0]
     [0 0 0 4]]
    '''
    
    in_diag = [[1,2],[3,4]] 
    print(sess.run(tf.diag_part(in_diag)))  #返回一个tensor的对角值[1,4]
    
    print(sess.run(tf.trace(in_diag)))  #求一个二维tensor的足迹，即对角值diagonal之和
    
    t = [[[1,2,3],[4,5,6],[7,8,9]],[[10,11,12],[13,14,15],[16,17,18]]]
    tt = tf.transpose(t,[2,1,0])    #让输入t，按照参数指定维度顺序进行转置操作，如果不设定参数，默认为一个全转置
    print(sess.run(tt))

    t1 = [[[[0,1,2,3],
            [4,5,6,7],
            [8,9,10,11]],
            [[12,13,14,15],
             [16,17,18,19],
             [20,21,22,23]]]]
    print(np.rank(t1))
    print(np.shape(t1))
    dim  = [3]
    rt = tf.reverse(t1,dim)     #沿着指定维度对输入进行反转，即指定维度的元素反转（元素内部不反转）
    print(sess.run(rt))
    dim  = [2]
    rt = tf.reverse(t1,dim)
    print(sess.run(rt))  
    
    dim  = [1]
    rt = tf.reverse(t1,dim)
    print(sess.run(rt)) 
    
    dims = [1,2]                #也可以按照多个维度反转
    rt = tf.reverse(t1,dims)
    print(sess.run(rt))
    
    t2 = [[1.,2.],[3.,4.]]
    t3 = [[5.,6.],[7.,8.]]
    mt = tf.matmul(t2,t3)  #矩阵相乘
    print(sess.run(mt))
    
    dt = tf.matrix_determinant(t2)      #返回矩阵的行列式
    print(sess.run(dt))
    
    it = tf.matrix_inverse(t2)      #求方阵的逆矩阵
    print(sess.run(it))
    print(sess.run(tf.matmul(it,t2)))
    
    #ct = tf.cholesky(t2)   #对输入方阵cholesky分解
    #print(sess.run(ct))    
    
#求解矩阵方程，返回矩阵变量。
#例如
# 2x +3y = 12
# x  + y= 5
sess = tf.InteractiveSession()
a = tf.constant([[2.,3.],[1.,1.]])
print(tf.matrix_solve(a,[[12.],[5.]]).eval())   #[[3.],[2.]] 即x = 3, y = 2
    
    
    
    
    
    

