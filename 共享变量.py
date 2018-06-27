# -*- coding: utf-8 -*-
"""
Created on Tue May 22 15:27:12 2018

@author: lWX379138
"""
import tensorflow as tf
import numpy as np

tf.reset_default_graph()

with tf.Session() as sess:
    
    var1 = tf.Variable(1.0,name="abfirstvar")
    print("var1:",var1.name)
    var1 = tf.Variable(2.0,name="abfirstvar")
    print("var1",var1.name)
    var2 = tf.Variable(3.0)
    print("var2:",var2.name)
    var2 = tf.Variable(4.0)
    print("var2:",var2.name)    
    
    sess.run(tf.global_variables_initializer())
    print("var1=",var1.eval())
    print("var2=",var2.eval())
    
    get_var1 = tf.get_variable("abfirstvar",[1],initializer = tf.constant_initializer(3.0))
    print("get_var1 name=",get_var1.name)
    
    get_var1 = tf.get_variable("abfirstvar1",[1],initializer = tf.constant_initializer(4.0))
    print("get_var1  name=",get_var1.name)
    
    #variable_scope 作用域
    with tf.variable_scope("test1",):#定义一个作用域test1
        var1 = tf.get_variable("abfirstvar",shape=[2],dtype=tf.float32)
        with tf.variable_scope("test2"):
            var2 = tf.get_variable("abfirstvar",shape=[2],dtype=tf.float32)
            pass
        pass
    with tf.variable_scope("test1",reuse = True):
        var3 = tf.get_variable("abfirstvar",shape=[2],dtype=tf.float32)
        with tf.variable_scope("test2"):
            var4 = tf.get_variable("abfirstvar",shape=[2],dtype=tf.float32)
            pass
        pass
    print("var1.name=",var1.name)
    print("var2.name=",var2.name)
    print("var3.name=",var3.name)
    print("var4.name=",var4.name)