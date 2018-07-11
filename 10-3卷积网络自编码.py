# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 11:05:42 2018

@author: lWX379138
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#导入MINST 数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/",one_hot=True)

learning_rate = 0.01    
n_conv_1 = 256        #第一层256个节点
n_conv_2 = 128        #第二层128个节点
n_input  = 784          #MNIST数据集图片的维度

#占位符
x = tf.placeholder("float",[None,n_input])      #输入
y = x                                           #输出

#学习参数
weights = {'encoder_conv1':tf.Variable(tf.truncated_normal([5,5,1,n_conv_1],stddev=0.1)),
           'encoder_conv2':tf.Variable(tf.truncated_normal([3,3,n_conv_1,n_conv_2],stddev=0.1)),
           'decoder_conv1':tf.Variable(tf.truncated_normal([5,5,1,n_conv_1],stddev=0.1)),
           'decoder_conv2':tf.Variable(tf.truncated_normal([3,3,n_conv_1,n_conv_2],stddev=0.1))
           }
biases = {'encoder_conv1':tf.Variable(tf.zeros([n_conv_1])),
          'encoder_conv2':tf.Variable(tf.zeros([n_conv_2])),
          
          'decoder_conv1':tf.Variable(tf.zeros([n_conv_1])),
          'decoder_conv2':tf.Variable(tf.zeros([n_input]))
          }

def conv2d(x,W):
    '''
    卷积运算 ，使用SAME填充方式   池化层后
         out_height = in_hight / strides_height（向上取整）
         out_width = in_width / strides_width（向上取整）
    
    args:
        x:输入图像 形状为[batch,in_height,in_width,in_channels] 
        W:权重 形状为[filter_height,filter_width,in_channels,out_channels]        
    '''
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

#编码
def encoder(x):
    h_conv1 = tf.nn.relu(conv2d(x,weights['encoder_conv1']) + biases['encoder_conv1'])
    h_conv2 = tf.nn.relu(conv2d(h_conv1,weights['encoder_conv2']) + biases['encoder_conv2'])
    return h_conv2,h_conv1
pass

#解码
def decoder(x,conv1):
    t_conv1 = tf.nn.conv2d_transpose(x-biases['decoder_conv2'],weights['decoder_conv2'],conv1.shape,[1,1,1,1])
    t_x_image = tf.nn.conv2d_transpose(t_conv1-biases['decoder_conv1'],weights['decoder_conv1'],x_image.shape,[1,1,1,1])
    return t_x_image
pass

'''
##max_pool_with_argmax只支持GPU操作
'''
def max_pool_with_argmax(net,stride):
    '''
    重定义一个最大池化函数，返回最大池化结果以及每个最大值的位置(是个索引，形状和池化结果一致) 'SAME'填充方式  池化层后
         out_height = in_hight / strides_height（向上取整）
         out_width = in_width / strides_width（向上取整）
    
    args:
        net:输入数据 形状为[batch,in_height,in_width,in_channels]
        stride：步长，是一个int32类型，注意在最大池化操作中我们设置窗口大小和步长大小是一样的
    '''
    #使用mask保存每个最大值的位置 这个函数只支持GPU操作
    _, mask = tf.nn.max_pool_with_argmax( net,ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1],padding='SAME')
    #将反向传播的mask梯度计算停止
    mask = tf.stop_gradient(mask)
    #计算最大池化操作
    net = tf.nn.max_pool(net, ksize=[1, stride, stride, 1],strides=[1, stride, stride, 1], padding='SAME') 
    #将池化结果和mask返回
    return net,mask

def un_max_pool(net,mask,stride):
    '''
    定义一个反最大池化的函数，找到mask最大的索引，将max的值填到指定位置
    args:
        net:最大池化后的输出，形状为[batch, height, width, in_channels]
        mask：位置索引组数组，形状和net一样
        stride:步长，是一个int32类型，这里就是max_pool_with_argmax传入的stride参数
    '''
    ksize = [1, stride, stride, 1]
    input_shape = net.get_shape().as_list()
    #  calculation new shape
    output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])
    # calculation indices for batch, height, width and feature maps
    one_like_mask = tf.ones_like(mask)
    batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int64), shape=[input_shape[0], 1, 1, 1])
    b = one_like_mask * batch_range
    y = mask // (output_shape[2] * output_shape[3])
    x = mask % (output_shape[2] * output_shape[3]) // output_shape[3]
    feature_range = tf.range(output_shape[3], dtype=tf.int64)
    f = one_like_mask * feature_range
    # transpose indices & reshape update values to one dimension
    updates_size = tf.size(net)
    indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
    values = tf.reshape(net, [updates_size])
    ret = tf.scatter_nd(indices, values, output_shape)
    return ret


x_image = tf.reshape(x,[-1,28,28,1])        #不要传入-1 否则会报错

#输出节点
encoder_out ,conv1= encoder(x_image)
h_pool2,mask = max_pool_with_argmax(encoder_out,2)

h_upool = un_max_pool(h_pool2,mask,2)
pred = decoder(h_upool,conv1)

#cost为y与pred的平方差
cost = tf.reduce_mean(tf.pow(y - pred , 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

#训练参数
training_epochs  = 200       #一共迭代20次
batch_size = 256            #每次取256个样本
display_step = 5            #迭代5次输出异常信息

#启动回话
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    total_batch = int(mnist.train.num_examples/batch_size)
    #开始训练
    for epoch in range(training_epochs):    #迭代
        for i in range(total_batch):
            batch_xs , batch_ys = mnist.train.next_batch(batch_size)    #取数据
            _ , c = sess.run([optimizer,cost] ,feed_dict={x:batch_xs})
            #训练模型
        if epoch % display_step == 0:       #显示日志信息
            print("Epoch :",'%04d'%(epoch + 1),"cost=","{:.9f}".format(c))
            pass

        pass
    print('完成')

    correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    #计算错误率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))
    print("Accuracy:",1-accuracy.eval({x:mnist.test.images,y:mnist.test.images}))
    
    print("Error:",cost.eval({x:batch_xs}))

    #可视化结果
    show_num = 10
    reconstruction = sess.run(pred,feed_dict={x:batch_xs})
    f , a = plt.subplots(2,10,figsize=(10,2))
    for i in range(show_num):
        a[0][i].imshow(np.reshape(mnist.test.images[i],(28,28)))
        a[1][i].imshow(np.reshape(reconstruction[i],(28,28)))
        pass
    plt.draw()
    pass
pass
