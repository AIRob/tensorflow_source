# -*- coding: utf-8 -*-
"""
Created on Tue May 22 10:52:55 2018

@author: lWX379138
"""
import tensorflow as tf
import numpy as np

with tf.Session() as sess:
    real = [2.25,3.25]
    imag = [4.75,5.75]
    com = tf.complex(real,imag)     #将两实数转换成复数形式
    print(sess.run(com))
    
    #x = [[-2.25+4.75j],[-3.25+5.75j]]
    #abs_x = tf.complex_abs(x)   #求复数的绝对值
    #print(sess.run(abs_x))

    #计算共轭复数 ,实数相等，复数相反
    ct = tf.conj(com)
    print(sess.run(ct))

    imag = tf.imag(com)
    real = tf.real(com)
    print(sess.run(real))
    print(sess.run(imag))
    
    fft = tf.fft(com)       #计算一维的离散傅里叶变换
    print(sess.run(fft))
    
    