# -*- coding: utf-8 -*-
"""
Created on Mon May 21 16:31:17 2018

@author: lWX379138
"""
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def moving_average(a,w=10):
    if len(a) < w:
        return a[:]
    return [ val if idx < w else sum(a[(idx-w):idx])/w for idx , val in enumerate(a)]        
    pass

train_X = np.linspace(-2,1,200)
train_Y = 2*train_X*train_X+np.random.randn(*train_X.shape)*0.3+(5+ np.random.randn(*train_X.shape)*0.1) #y=2x ,并加入噪声
#显示模拟噪声
plt.plot(train_X,train_Y,'ro',label='Orginal data')
plt.legend()
plt.show()

#重置图
tf.reset_default_graph()

#创建模型
#占位符
X= tf.placeholder("float")      #x代表输入
Y = tf.placeholder("float")     #y 代表真实值

#模型参数
W = tf.Variable(tf.random_normal([1]),name="weight")    #tf.Variable变量
b = tf.Variable(tf.zeros([1]),name='bias')  

#前向结构
z = tf.multiply(tf.multiply(X,X),W) +b         # tf.multiply代表相乘，+b 就等于z了
tf.summary.histogram('z',z)                     #将预测值以直方图形式显示

#'''
#反向优化
cost = tf.reduce_mean(tf.square(Y-z))   #生成值和真实值得平方差
tf.summary.scalar("loss_function",cost) #将损失以标量形式显示
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#初始化所有变量
init = tf.global_variables_initializer()
#定义参数
training_epochs = 20
display_step = 2


#启动session
with tf.Session() as sess:
    sess.run(init)
    
    merged_summary_op = tf.summary.merge_all() #合并所有 summary
    #创建 summary_writer ,用于写文件
    summary_writer = tf.summary.FileWriter("log/mnist_with_summaries",sess.graph)
    
    
    plotdata = {'batchsize':[],'loss':[]}   #存放批次值和损失值
    #向模型输入数据
    for epoch in range(training_epochs):
        for(x,y) in zip(train_X,train_Y):
            sess.run(optimizer,feed_dict={X:x,Y:y})
            pass
        #生成summary
        summary_str = sess.run(merged_summary_op,feed_dict={X:x,Y:y})
        summary_writer.add_summary(summary_str,epoch)#将summary 写入文件
        
        #显示孙连中的详细信息
        if epoch % display_step == 0:
            loss = sess.run(cost,feed_dict={X:train_X,Y:train_Y})
            print("Epoch :",epoch+1,"cost=",loss,"W=",sess.run(W),"b=",sess.run(b))
            if not (loss=="NA"):
                plotdata["batchsize"].append(epoch)
                plotdata["loss"].append(loss)
                pass
            pass
        pass
    print("Finished!")
    #保存模型
    saver = tf.train.Saver()    # saver = tf.train.Saver({'weight':W,‘bias’:b}) #saver = tf.train.Saver([W,b])
    savedir = "E:/TensorFlow/mySrc/lianxi/log/"
    saver.save(sess,savedir+"linermoder.cpkt")
    print("cost=",sess.run(cost,feed_dict={X:train_X,Y:train_Y}),"W=",sess.run(W),"b=",sess.run(b))
    #图形显示
    plt.plot(train_X,train_Y,'ro',label='Original data')
    plt.plot(train_X,sess.run(W)*train_X*train_X+sess.run(b),label='Fittendline')
    plt.legend()
    plt.show()
    
    plotdata['avgloss'] = moving_average(plotdata['loss'])
    plt.figure(1)
    #plt.subplot(711)
    plt.plot(plotdata['batchsize'],plotdata['avgloss'],'b--')
    plt.xlabel('Minibatch number')
    plt.ylabel("Loss")
    plt.title("Minibatch run vs . Training loss")
    plt.show()

    print("x=0.5,z=",sess.run(z,feed_dict={X:0.5}))

'''

#使用模型
with tf.Session() as sess2:
    init = tf.global_variables_initializer()
    sess2.run(init)
    #打开之前保存的模型
    saver = tf.train.Saver()
    savedir = "E:/TensorFlow/mySrc/lianxi/log/"
    saver.restore(sess2,savedir+"linermoder.cpkt")
    print("x=2,z=",sess2.run(z,feed_dict={X:2}))
    pass

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
savedir = "E:/TensorFlow/mySrc/lianxi/log/"
print_tensors_in_checkpoint_file(savedir+'linermoder.cpkt',None,True,True)

'''


