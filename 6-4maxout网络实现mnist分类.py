# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 11:33:52 2018

@author: lWX379138
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot = True)
import pylab

tf.reset_default_graph()
#定义占位符
x = tf.placeholder(tf.float32,[None,784])   #MNIST 数据集的维度是28x28=784
y = tf.placeholder(tf.float32,[None,10])    #数字0~9，共10个类别

W = tf.Variable(tf.random_normal([784,10]))
b = tf.Variable(tf.zeros([10]))

z = tf.matmul(x,W) + b
maxout = tf.reduce_max(z,axis=1,keep_dims = True)
#设置学习参数
W2 = tf.Variable(tf.truncated_normal([1,10],stddev=0.1))
b2 = tf.Variable(tf.zeros([1]))


pred = tf.nn.softmax(tf.matmul(x,W) +b)     #softmax 分类
#pred = tf.nn.softmax(tf.matmul(maxout,W2) +b2)     #softmax 分类

#损失函数 交叉熵 P107页，最后一行
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),reduction_indices = 1)) 
#等价于 # 
#cost = tf.nn.softmax_cross_entropy_with_logits(labels = y,logits=pred)
#spare 交叉熵 P110页
#cost = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y,logits = pred)

#定义参数
learning_rate = 0.01 #0.01
#使用梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
training_epochs = 200
batch_size = 100
display_step = 1

#'''
#启动session 1 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) #initializing OP
    
    #启动循环训练任务
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        #循环所有数据集
        for i in range(total_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            #运行优化器
            _, c = sess.run([optimizer,cost],feed_dict={x:batch_xs,y:batch_ys})
            #计算平均值loss
            avg_cost += c/total_batch
            pass
        #显示训练中的详细信息
        if (epoch +1 ) % display_step == 0:
            print("Epoch:","%04d"%(epoch+1),"cost=","{:.9f}".format(avg_cost))
            pass
        pass
    
    print("train Finished!")
    
    #测试model 校正预测
    correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    #计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    print("Accuracy:",accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))
    
    #保存模型
    saver = tf.train.Saver()
    model_path = "log/521model.ckpt"
    save_path = saver.save(sess,model_path)
    print("Model saved in file:%s"%save_path)
#'''

'''
print("Starting 2nd session...")
with tf.Session() as sess:
    #初始化变量
    sess.run(tf.global_variables_initializer())
    #恢复模型变量
    #保存模型
    saver = tf.train.Saver()
    model_path = "log/521model.ckpt"
    saver.restore(sess,model_path)
    
    #测试model
    correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    #计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    print("Accuracy:",accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))
    
    output = tf.argmax(pred,1)
    batch_xs,batch_ys = mnist.train.next_batch(4)
    outputval,predv = sess.run([output,pred],feed_dict={x:batch_xs,y:batch_ys})
    print(outputval,predv)
    print(batch_xs.shape,batch_ys)
    
    im = batch_xs[0]
    im = im.reshape(-1,28)
    pylab.imshow(im)
    pylab.show()

    im = batch_xs[1]
    im = im.reshape(-1,28)
    pylab.imshow(im)
    pylab.show()
    
    im = batch_xs[2]
    im = im.reshape(-1,28)
    pylab.imshow(im)
    pylab.show()
    
    im = batch_xs[3]
    im = im.reshape(-1,28)
    pylab.imshow(im)
    pylab.show()

    print("End of 2nd session !")    
#'''    
    