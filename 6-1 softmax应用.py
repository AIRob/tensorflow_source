# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 18:55:20 2018

@author: lWX379138
"""
import tensorflow as tf

labels = [[0,0,1],[0,1,0]]
labels2 = [[0.4,0.1,0.5],[0.3,0.6,0.1]]
#spare 标签
labels3 = [2,1] #表明labels3 中共有3个分类：0,1,2. [2,1]等价于onehot 编码中的 0 0 1 和 0 1 0



logits = [[2,0.5,6],[0.1,0,3]]

logits_scaled = tf.nn.softmax(logits)
logits_scaled2 = tf.nn.softmax(logits_scaled)

result1 = tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits)
result2 = tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits_scaled)
result3 = -tf.reduce_sum(labels*tf.log(logits_scaled),1)
result4 = tf.nn.softmax_cross_entropy_with_logits(labels=labels2,logits=logits)
result5 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels3,logits=logits)

loss = tf.reduce_sum(result1)
loss1 = -tf.reduce_sum(labels*tf.log(logits_scaled))
loss2 = tf.reduce_sum(-tf.reduce_sum(labels*tf.log(logits_scaled),1))

with tf.Session() as sess:
    print("scaled=",sess.run(logits_scaled))
    print("scaled2=",sess.run(logits_scaled2))
    
    print("rell=",sess.run(result1),"\n") #正确的方式
    print("rel2=",sess.run(result2),"\n") #如果将softmax变换完的值放进去，就相当于算第二次softmax的loss，所以会出错
    print("rel3=",sess.run(result3),"\n")
    print("rel4=",sess.run(result4),"\n")
    print("rel5=",sess.run(result5),"\n")
    
    print("loss=",sess.run(loss),"\n")
    print("loss1=",sess.run(loss1),"\n")
    print("loss2=",sess.run(loss2),"\n")
    
    print("sess Finished !")
