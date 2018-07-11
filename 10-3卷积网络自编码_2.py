# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 14:10:49 2018
部分参考：https://blog.csdn.net/qq_34464926/article/details/80936150
@author: lWX379138
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('data/', validation_size=0)

img = mnist.train.images[2]
plt.imshow(img.reshape((28, 28)), cmap='Greys_r')

learning_rate = 0.001
# Input and target placeholders
inputs_ = tf.placeholder(tf.float32, (None, 28,28,1), name="input")
targets_ = tf.placeholder(tf.float32, (None, 28,28,1), name="target")

### 编码器--压缩
conv1 = tf.layers.conv2d(inputs_, 16, (3,3), padding='same', activation=tf.nn.relu)
# 当前shape: 28x28x16
maxpool1 = tf.layers.max_pooling2d(conv1, (2,2), (2,2), padding='same')
# 当前shape: 14x14x16
conv2 = tf.layers.conv2d(maxpool1, 8, (3,3), padding='same', activation=tf.nn.relu)
# 当前shape: 14x14x8
maxpool2 = tf.layers.max_pooling2d(conv2, (2,2), (2,2), padding='same')
# 当前shape: 7x7x8
conv3 = tf.layers.conv2d(maxpool2, 8, (3,3), padding='same', activation=tf.nn.relu)
# 当前shape: 7x7x8
encoded = tf.layers.max_pooling2d(conv3, (2,2), (2,2), padding='same')
# 当前shape: 4x4x8

### 解码器--还原
upsample1 = tf.image.resize_nearest_neighbor(encoded, (7,7))
# 当前shape: 7x7x8
conv4 = tf.layers.conv2d(upsample1, 8, (3,3), padding='same', activation=tf.nn.relu)
# 当前shape: 7x7x8
upsample2 = tf.image.resize_nearest_neighbor(conv4, (14,14))
# 当前shape: 14x14x8
conv5 = tf.layers.conv2d(upsample2, 8, (3,3), padding='same', activation=tf.nn.relu)
# 当前shape: 14x14x8
upsample3 = tf.image.resize_nearest_neighbor(conv5, (28,28))
# 当前shape: 28x28x8
conv6 = tf.layers.conv2d(upsample3, 16, (3,3), padding='same', activation=tf.nn.relu)
# 当前shape: 28x28x16


logits = tf.layers.conv2d(conv6, 1, (3,3), padding='same', activation=None)
#当前shape: 28x28x1


decoded = tf.nn.sigmoid(logits, name='decoded')


#计算损失函数
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=logits)
cost = tf.reduce_mean(loss)
#使用adam优化器优化损失函数
opt = tf.train.AdamOptimizer(0.001).minimize(cost)


#训练网络
sess = tf.Session()
epochs = 20
batch_size = 200
sess.run(tf.global_variables_initializer())
for e in range(epochs):
    for ii in range(mnist.train.num_examples//batch_size):
        batch = mnist.train.next_batch(batch_size)
        imgs = batch[0].reshape((-1, 28, 28, 1))
        batch_cost, _ = sess.run([cost, opt], feed_dict={inputs_: imgs,
                                                         targets_: imgs})


        print("Epoch: {}/{}...".format(e+1, epochs),
              "Training loss: {:.4f}".format(batch_cost))

#matplotlib绘图查看压缩后还原的图片与原图片的区别
fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(20,4))
in_imgs = mnist.test.images[:10]
reconstructed = sess.run(decoded, feed_dict={inputs_: in_imgs.reshape((10, 28, 28, 1))})


for images, row in zip([in_imgs, reconstructed], axes):
    for img, ax in zip(images, row):
        ax.imshow(img.reshape((28, 28)), cmap='Greys_r')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        
fig.tight_layout(pad=0.1)

sess.close()













