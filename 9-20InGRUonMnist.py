# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 14:46:23 2018

@author: lWX379138
"""
from tensorflow.python.ops.rnn_cell_impl import  RNNCell
from tensorflow.python.ops.math_ops import sigmoid,tanh
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import _linear

import tensorflow as tf
print(tf.__version__)
#导入MINST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('data/',one_hot=True)
tf.reset_default_graph()

def ln(tensor,scope=None,epsilon = 1e-5):
    '''Layer normalizers a 2D tensor along its second axis '''
    assert(len(tensor.get_shape()) == 2)
    m,v = tf.nn.moments(tensor,[1],keep_dims = True)
    if not isinstance(scope,str):
        scope = ''
        pass
    with tf.variable_scope(scope + 'layer_norm'):
        scale = tf.get_variable('scale',
                                shape = [tensor.get_shape()[1]],
                                initializer = tf.constant_initializer(1) )
        shift = tf.get_variable('shift',
                                shape = [tensor.get_shape()[1]],
                                initializer = tf.constant_initializer(0) )
        pass
    LN_initial = (tensor - m) / tf.sqrt(v + epsilon)
    
    return LN_initial * scale + shift
pass

class LNGRUCell(RNNCell):
    '''Gated Recurrent Unit cell(cf . http://arxiv.org/abs/1406.1078). '''
    
    def __init__(self,num_units,input_size= None,activation=tanh):
        if input_size is not None:
            print('%s:The input_size parameter is deprecated .'%self)
            pass
        self._num_units = num_units
        self._activation = activation
        pass
    
    @property
    def state_size(self):
        return self._num_units
        pass
    
    @property
    def output_size(self):
        return self._num_units
        pass
    
    def __call__(self,inputs,state):
        '''Gated recurrent unit (GRU) with nunits cells'''
        with vs.variable_scope('Gates'):
            value = _linear([inputs,state],2*self._num_units,True,kernel_initializer=tf.constant_initializer(1.0))
            r,u = array_ops.split(value = value,num_or_size_splits = 2,axis = 1)
            r = ln(r,scope = 'r/')
            u = ln(u,scope = 'u/')
            r,u = sigmoid(r),sigmoid(u)
            pass
        with vs.variable_scope('Candidate'):
            Cand = _linear([inputs,r*state],self._num_units,True)
            c_pre = ln(Cand,scope='new_h/')
            c = self._activation(c_pre)
            pass
        new_h = u*state + (1-u)*c
        return new_h,new_h
    pass
pass


n_input = 28                #MNIST data 输入（img shape :28*28)
n_steps = 28                #序列个数
n_hidden = 128              #隐藏层个数
n_classes = 10              #MNIST 分类个数 （0~9 digits）

#定义占位符
x = tf.placeholder('float',[None,n_steps,n_input])
y = tf.placeholder('float',[None,n_classes])

x1 = tf.unstack(x,n_steps,1)

gru = LNGRUCell(n_hidden)
outputs ,states = tf.contrib.rnn.static_rnn(gru,x1,dtype = tf.float32)
pred = tf.contrib.layers.fully_connected(outputs[-1],n_classes,activation_fn = None)
#定义参数
learning_rate = 0.01
batch_size = 128
time_steps = 28
#损失函数 交叉熵 P107页，最后一行
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))#tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),reduction_indices = 1)) 
#等价于 
#cost = tf.nn.softmax_cross_entropy_with_logits(labels = y,logits=pred)

#使用梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
#测试model 校正预测
correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#'''
#启动session 1 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) #initializing OP
    
    iter=1
    while iter<800:
        batch_x,batch_y=mnist.train.next_batch(batch_size=batch_size)
        batch_x=batch_x.reshape((batch_size,time_steps,n_input))
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if iter %100==0:
            acc=sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
            los=sess.run(cost,feed_dict={x:batch_x,y:batch_y})
            print("For iter ",iter)
            print("Accuracy ",acc)
            print("Loss ",los)
            print("__________________")
        iter=iter+1
    #calculating test accuracy
    test_data = mnist.test.images[:128].reshape((-1, time_steps, n_input))
    test_label = mnist.test.labels[:128]
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))    
    
