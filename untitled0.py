# -*- coding: utf-8 -*-
"""
Created on Fri May 18 09:12:53 2018

@author: lWX379138
"""

import tensorflow as tf
hello = tf.constant("hello,tensorflow")
sess = tf.Session()
print(sess.run(hello))
sess.close()
