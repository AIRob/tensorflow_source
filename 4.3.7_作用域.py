# -*- coding: utf-8 -*-
"""
Created on Tue May 22 16:41:22 2018

@author: lWX379138
"""
import tensorflow as tf
import numpy as np

tf.reset_default_graph()

with tf.variable_scope("scope1") as sp:
    var1 = tf.get_variable("v",[1])
    
print("sp:",sp.name)
print("var1:",var1.name)

with tf.variable_scope("scope2"):
    var2 = tf.get_variable("v",[1])
    
    with tf.variable_scope(sp) as sp1:
        var3 = tf.get_variable("v3",[1])
        with tf.variable_scope(""):
            var4 = tf.get_variable("v4",[1])
            pass
        pass
    pass

print("sp1:",sp1.name)
print("var2:",var2.name)
print("var3:",var3.name)
print("var4:",var4.name)
with tf.variable_scope("scope"):
    with tf.name_scope("bar"):
        v = tf.get_variable("v",[1])
        x = 1.0 + v
        with tf.name_scope(""):
            y = 2.0 +v
        pass
    pass

print("v:",v.name)
print("x.op:",x.op.name)
print("y.op:",y.op.name)


        