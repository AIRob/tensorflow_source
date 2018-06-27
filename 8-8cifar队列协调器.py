# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 10:56:19 2018

@author: lWX379138
"""
import cifar10_input
import tensorflow as tf
import numpy as np
import pylab


#取数据 目录：cifar-10-batches-py
batch_size = 128 
data_dir = 'cifar-10-binary/cifar-10-batches-bin'
images_test,labels_test = cifar10_input.inputs(eval_data = True, data_dir = data_dir, batch_size = batch_size)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.train.start_queue_runners()      #运行队列 ，大数据常用方法
image_batch , label_batch = sess.run([images_test,labels_test])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    #定义协调器
    coord = tf.train.Coordinator()
    
    #启动入队线程，coordinator 是 线程参数
    threads = tf.train.start_queue_runners(sess,coord)
    
    image_batch,label_batch = sess.run([images_test,labels_test])
    
    print("_\n",image_batch[0])
    
    print("_\n",label_batch[0])
    
    pylab.imshow(image_batch[0])
    pylab.show()
    
    coord.request_stop()
        
        

