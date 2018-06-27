# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 17:05:06 2018

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

print("_\n",image_batch[0])
print("_\n",label_batch[0])
pylab.imshow(image_batch[0])
pylab.show()