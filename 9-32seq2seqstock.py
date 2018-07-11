# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 09:22:35 2018

@author: lWX379138
"""
import random
import math

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pands as pd
pd.options.mode.chained_assignment = None   #default = 'warn'

csv_fn = '600000.csv'

#载入股票函数
def loadstock(window_size):
    names = ['date','code','name','Close','top_price','low_price','opening_price','bef_price','floor_price','floor','exchange','Volume','amount','总市值','流通市值']
    data = pd.read_csv(csv_fn,names=names,header = None,encoding='gbk')
    
    predictor_names = ['Close']
    training_features = np.asarray(data[predictor_names],dtype = 'float32')
    kept_values = training_features[1000:]
    
    X = []
    Y = []
    for i in range(len(kept_values) - window_size * 2):
        #X为前window_size 个序列，Y 为后window_size 个 序列
        X.append(kept_values[i:i+window_size])
        Y.append(kept_values[i+window_size:i+window_size*2])
        pass
    
    X = np.reshape(X,[-1,window_size,len(predictor_names)])
    Y = np.reshape(Y,[-1,window_size,len(predictor_names)])
    print(X.shape)
    
    return X,Y
pass

def do_generate_x_y(X,Y,batch_size):
    assert X.shape == Y.shape,(X.shape,Y.shape)
    idxes = np.random.randint(X.shape[0],size = batch_size)
    X_out = np.array(X[idxes]).transpose((1,0,2))
    Y_out = np.array(Y[idxes]).transpose((1,0,2))
    return X_out,Y_out
pass

#生成 15 个连续序列，将cos 和 sin 随机偏移变化后的值叠加起来
def generate_data(isTrain,batch_size):
    #用前40个样本来预测后40个样本
    
    seq_length = 40
    seq_length_test = 40
    
    global Y_train
    global X_train
    global X_test
    global Y_test
    
    #载入内存
    if len(Y_train) == 0:
        X,Y = loadstock(window_size=seq_length)
        
        #split 80-20
        X_train = X[:int(len(X) * 0.8)]
        Y_train = Y[:int(len(Y) * 0.8)]
        pass
    
    if len(Y_test) == 0:
        X,Y = loadstock( window_size= seq_length_test)
        
        #split 80 -20
        X_test = X[int(len(X) * 0.8 ):]
        Y_test = Y[int(len(Y) * 0.8 ):]
        pass
    
    
    if isTrain:
        return do_generate_x_y(X_train,Y_train,batch_size)
    else:
        return do_generate_x_y(X_test,Y_test,batch_size)
    pass
pass

sample_now ,sample_f = generate_data(isTrain= True,batch_size=3)
print('training examples:')
print(sample_now.shape)
print('(seq_length,batch_size,output_dim)')

seq_length = sample_now.shape[0]
batch_size = 100        #股票数据没有固定规则，并且数据量大，所以加大batch_size 

output_dim = input_dim = sample_now.shape[-1]
hidden_dim = 12
layers_stacked_count = 2

#学习率
learning_rate = 0.04
nb_iters = 100000           #加大迭代次数

lambda_12_reg = 0.003       #L2 正则参数

tf.reset_default_graph()

encoder_input = []
expected_output = []
decode_input = []

for i in range(seq_length):
    encoder_input.append(tf.placeholder(tf.float32,shape=(None,input_dim)) )
    expected_output.append(tf.placeholder(tf.float32,shape=(None,output_dim)) )
    decode_input.append(tf.placeholder(tf.float32,shape=(None,input_dim)) )
    pass

tcells = []
for i in range(layers_stacked_count):
    tcells.append(tf.contrib.rnn.GRUCell(hidden_dim))
    pass
Mcell = tf.contrib.rnn.MultiRNNCell(tcells)

dec_outputs,dec_memory = tf.contrib.legacy_seq2seq.basic_rnn_seq2seq(encoder_input,decode_input,Mcell)

reshaped_outputs = []
for ii in dec_outputs:
    reshaped_outputs.append(tf.contrib.layers.fully_connected(ii,output_dim,activation_fn = None) )
    pass

#计算 L2 的 loss 值
output_loss = 0
for _y,_Y in zip(reshaped_outputs,expected_output):
    output_loss += tf.reduce_mean( tf.pow (_y - _Y , 2 ) )
    pass

#求正则化loss值
reg_loss = 0
for tf_var in tf.trainable_variables():
    if not ('fully_connected' in tf_var.name ):
        reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))
        pass
    pass

loss = output_loss + lambda_12_reg * reg_loss
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
sess = tf.InteractiveSession()

def train_batch(batch_size):
    X, Y = generate_data(isTrain=True,batch_size=batch_size)
    feed_dict = {encoder_input[t] : X[t] for t in range(len(encoder_input))}
    feed_dict.update({expected_output[t]:Y[t] for t in range(len(expected_output))})
    
    c = np.concatenate(([np.zeros_like(Y[0])] , Y[:-1] ),axis = 0)
    
    feed_dict.update({decode_input[t]:c[t] for t in range(len(c)) })
    
    _,loss_t = sess.run([train_op,loss],feed_dict)
    return loss_t
pass

def test_batch(batch_size):
    X, Y = generate_data(isTrain=True,batch_size=batch_size)
    feed_dict = {encoder_input[t] : X[t] for t in range(len(encoder_input))}
    feed_dict.update({expected_output[t]:Y[t] for t in range(len(expected_output))})
    
    c = np.concatenate(([np.zeros_like(Y[0])] , Y[:-1] ),axis = 0)
    
    feed_dict.update({decode_input[t]:c[t] for t in range(len(c)) })
    output_lossv ,reg_lossv ,loss_t = sess.run([output_loss,reg_loss,loss],feed_dict)
    print('--------------------')
    print(output_lossv,reg_lossv)
    return loss_t
pass

#训练
train_losses = []
test_losses = []

sess.run(tf.global_variables_initializer())

for t in range(nb_iters + 1):
    train_loss = train_batch(batch_size)
    train_losses.append(train_loss)
    if t % 50 == 0:
        test_loss = test_batch(batch_size)
        test_losses.append(test_loss)
        print("Step {}/{} ,train loss :{},\tTEST loss:{}".format(t,nb_iters,train_loss,test_loss))
        pass
    pass

print("Fin.train loss: {},\tTEST loss :{}".format(train_loss,test_loss))

#输出loss 图例
plt.figure(figsize = (12,6))
plt.plot(np.array(range(0,len(test_losses))) / float(len(test_losses) -1) * (len(train_losses) -1 ),np.log(test_losses),label="Test loss" )
plt.plot(np.log(train_losses) , label = "Train loss")
plt.title("Training errors over time(on a logarithmic scale)")
plt.xlabel("Iteration")
plt.ylabel("log(LOSS)")
plt.legend(loc='best')
plt.show()

#测试
nb_predictions = 4
print("visualize {} predictions data:".format(nb_predictions))

preout = []
X,Y = generate_data(isTrain=False,batch_size= nb_predictions)
print(np.shape(X),np.shape(Y))
for tt in range(seq_length):
    feed_dict = {encoder_input[t]:X[t+tt] for t in range(seq_length)}
    feed_dict.update({expected_output[t] :Y[t+tt] for t in range(len(expected_output))})
    c = np.concatenate(([np.zeros_like(Y[0])],Y[tt:seq_length+tt-1]),axis = 0)      #从前15个序列的最后一个开始预测
    
    feed_dict.update({decode_input[t] :c[t] for t in range(len(c))})
    outputs = np.array(sess.run([reshaped_outputs],feed_dict)[0] )
    preout.append(outputs[-1])
    
    pass
print(np.shape(preout))  #将每个未知预测值收集起来准备显示出来
preout = np.reshape(preout,[seq_length,nb_predictions,output_dim])

for j in range(nb_predictions):
    plt.figure(figsize=(12,3))
    
    for k in range(output_dim):
        past = X[:,j,k]
        expected = Y[seq_length-1:,j,k]
        
        pred = preout[:,j,k]
        
        label1 = "past" if k == 0 else "_nolegend_"
        label2 = 'future' if k ==0 else '_nolegend_'
        label3 = 'Pred' if k ==0 else '_nolegend_'
        plt.plot(range(len(past)),past,'o--b',label = label1)
        plt.plot(range(len(past),len(expected) + len(past)),expected,'x--b',label =label2)
        plt.plot(range(len(past),len(pred) + len(past)),pred,'o--y',label = label3)
        pass
    
    plt.legend(loc = 'best')
    plt.title("Predictions vs. future")
    plt.show()
    
    pass
pass


