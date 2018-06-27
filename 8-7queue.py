# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 10:15:23 2018

@author: lWX379138
"""

import tensorflow as tf

#创建长度为 100 的队列
queue = tf.FIFOQueue(15,'float')

c = tf.Variable(0.0)        #计数器
#加 1操作
op = tf.assign_add(c,tf.constant(1.0))
enqueue_op = queue.enqueue(c)

#创建一个队列管理器 QueueRunner , 用这两个操作向q 中添加元素。目前我们只使用一个线程
qr = tf.train.QueueRunner(queue,enqueue_ops = [op,enqueue_op])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    coord = tf.train.Coordinator()
    
    #启动入队线程，coordinator 是 线程参数
    enqueue_threads = qr.create_threads(sess,coord=coord,start= True)
    
    #主线程
    for i in range(0,10):
        print("-----------------------")
        print(sess.run(queue.dequeue())) 
        pass
    
    coord.request_stop() #通知其他线程关闭，其他线程关闭后，这一函数才能返回
    
    ##指定等待某个进程结束
    #coord.join(enqueue_threads)
    #print(sess.run(queue.dequeue())) 
        
        