# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 10:44:10 2018

@author: lWX379138
"""
import tensorflow as tf
from tensorflow.contrib import rnn
#导入MINST data数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('data/',one_hot=True)

#定义参数
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

#网络模型参数设置
n_input = 28                #MNIST data 输入（img shape ：28 * 28 ）
n_steps = 28                #序列个数
n_hidden = 128              #隐藏层节点个数
n_classes = 10              #MNIST 分类数 （0 ~  9 digits )

tf.reset_default_graph()

#定义占位符
x = tf.placeholder('float',[None,n_steps,n_input])
y = tf.placeholder('float',[None,n_classes])

x1 = tf.unstack(x,n_steps,1)
lstm_fw_cell = rnn.BasicLSTMCell(n_hidden,forget_bias = 1.0)
#反向cell
lstm_bw_cell = rnn.BasicLSTMCell(n_hidden,forget_bias = 1.0)
outputs,output_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell,x,dtype = tf.float32)

print(len(outputs),outputs[0].shape,outputs[1].shape)
outputs = tf.concat(outputs,2)
outputs = tf.transpose(outputs,[1,0,2])

pred = tf.contrib.layers.fully_connected(outputs[-1],n_classes,activation_fn = None)


