# -*- coding: utf-8 -*-
"""
Created on Mon May 21 19:26:21 2018

@author: lWX379138
"""

import tensorflow as tf
import numpy as np

t = [[[1,1,1],[2,2,2]],[[3,3,3],[4,4,4]],[[5,5,5],[6,6,6]]]
slicet1 = tf.slice(t,[0,0,0],[1,1,3])


print(slicet1)


x = tf.constant(2)
y = tf.constant(5)

def f1():
    return tf.multiply(x,17)
    pass
def f2():
    return tf.add(y,23)
    pass


with tf.Session() as sess:
    print(sess.run(slicet1))
    print(sess.run(f1()))
    print(sess.run(f2()))
