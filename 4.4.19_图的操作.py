# -*- coding: utf-8 -*-
"""
Created on Tue May 22 17:18:26 2018

@author: lWX379138
"""
import numpy as np
import tensorflow as tf

c= tf.constant(0.0)

g= tf.Graph()   #建立图
with g.as_default():    
    c1 = tf.constant(0.0)
    print("c1.graph")
    print(g)
    print(c.graph)
    
g2 = tf.get_default_graph() #获取默认图
print(g2)

tf.reset_default_graph()    #重置图
g3 = tf.get_default_graph() #获取默认图
print(g3)
#获取张量 tensor
print(c1.name)
t = g.get_tensor_by_name(name="Const:0")
print(t)

print("获取节点操作...")
#获取节点操作
a = tf.constant([[1.0,2.0]])
b = tf.constant([[1.0],[3.0]])

tensor1 = tf.matmul(a,b,name="exampleop")
print("tensor1.name:",tensor1.name,' tensor1:',tensor1)
test = g3.get_tensor_by_name("exampleop:0")
print('test:',test)

print(tensor1.op.name)
testop = g3.get_operation_by_name("exampleop")
print('testop:',testop)

with tf.Session() as sess:
    test = sess.run(test)
    print('test:',test)
    test = tf.get_default_graph().get_tensor_by_name("exampleop:0")
    print('test:',test)
    
    tt2 = g.get_operations()
    print(tt2)
    
    tt3 = g.as_graph_element(c1)
    print(tt3)

