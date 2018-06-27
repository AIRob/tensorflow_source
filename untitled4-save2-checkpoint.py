# -*- coding: utf-8 -*-
"""
Created on Mon May 21 15:57:55 2018

@author: lWX379138
"""
import tensorflow as tf
tf.reset_default_graph()
global_step = tf.train.get_or_create_global_step()
step = tf.assign_add(global_step,1)
#设置检查点路径 为 log/checkpoints
with tf.train.MonitoredTrainingSession(checkpoint_dir="log/checkpoints",save_checkpoint_secs = 2 ) as sess:
    print(sess.run([global_step]))
    while not sess.should_stop():       #启动死循环，当sess 不结束时就不停止
        i = sess.run(step)
        print(i)
        pass
    pass



