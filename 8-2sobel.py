# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 15:51:46 2018

@author: lWX379138
"""
import matplotlib.pyplot as plt   #plt用来显示图片
import matplotlib.image as mpimg  #mpimg 用来读取图片
import numpy as np
import tensorflow as tf

myimg = mpimg.imread('img.jpg')     #读取图片
plt.imshow(myimg)   #显示图片
plt.axis('off')     #不显示坐标轴
plt.show()

#(图片高，图片宽，通道数)
print(myimg.shape)

full = np.reshape(myimg,[1,2448,3264,3])
inputfull = tf.Variable(tf.constant(1.0,shape=[1,2448,3264,3]))

filter = tf.Variable(tf.constant([[-1.0,-1.0,-1.0],[0,0,0],[1.0,1.0,1.0],
                                  [-2.0,-2.0,-2.0],[0,0,0],[2.0,2.0,2.0],
                                  [-1.0,-1.0,-1.0],[0,0,0],[1.0,1.0,1.0]],
                                  shape = [3,3,3,1]))
'''                                  
#4x4                                  
filter = tf.Variable(tf.constant([[-1.0,-1.0,-1.0,-1.0],[0,0,0,0],[1.0,1.0,1.0,1.0],
                                  [-2.0,-2.0,-2.0,-2.0],[0,0,0,0],[2.0,2.0,2.0,2.0],
                                  [-2.0,-2.0,-2.0,-2.0],[0,0,0,0],[2.0,2.0,2.0,2.0],
                                  [-1.0,-1.0,-1.0,-1.0],[0,0,0,0],[1.0,1.0,1.0,1.0]],
                                  shape = [4,4,3,1]))
#5x5
filter = tf.Variable(tf.constant([[-1.0,-1.0,-1.0,-1.0,-1.0],[0,0,0,0,0],[1.0,1.0,1.0,1.0,1.0],
                                  [-2.0,-2.0,-2.0,-2.0,-2.0],[0,0,0,0,0],[2.0,2.0,2.0,2.0,2.0],
                                  [-3.0,-3.0,-3.0,-3.0,-3.0],[0,0,0,0,0],[3.0,3.0,3.0,3.0,3.0],
                                  [-2.0,-2.0,-2.0,-2.0,-2.0],[0,0,0,0,0],[2.0,2.0,2.0,2.0,2.0],
                                  [-1.0,-1.0,-1.0,-1.0,-1.0],[0,0,0,0,0],[1.0,1.0,1.0,1.0,1.0]],
                                  shape = [5,5,3,1]))
#'''                                  
                                  
op = tf.nn.conv2d(inputfull,filter,strides=[1,1,1,1],padding='SAME')     #'SAME' 边缘填充
#3个通道，生成1个feature map
o = tf.cast( ( (op-tf.reduce_min(op))/(tf.reduce_max(op) - tf.reduce_min(op) ) )*255,tf.uint8 ) #归一化 255

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer() )
    t,f = sess.run([o,filter],feed_dict={inputfull:full})
    
    
    t = np.reshape(t,[2448,3264])
    
    plt.imshow(t,cmap="Greys_r")
    plt.axis("off")
    plt.show()

