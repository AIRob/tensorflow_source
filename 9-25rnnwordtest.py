# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 18:38:34 2018

@author: lWX379138
"""
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random,time
from collections import Counter

start_time  = time.time()

def elapsed(sec):
    if sec <60:
        return str(sec)+' sec'
    elif sec <(60*60):
        return str(sec/60) +" min"
    else:
        return str(sec/(60*60)) + ' hr'
    pass
pass

tf.reset_default_graph()

training_file = 'wordstest.txt'

#出来多个中文文件
def readalltxt(txt_files):
    labels = []
    for txt_file in txt_files:
        target = get_ch_lable(txt_file)
        labels.append(target)
        pass
    return labels
pass

#处理汉字
def get_ch_lable(txt_file):
    labels = ""
    with open(txt_file,"rb") as f:
        for label in f:
            labels = labels + label.decode('utf-8')
            pass
        pass
    return labels
pass

#优先转文件里的字符到向量
def get_ch_lable_v(txt_file,word_num_map,txt_label= None):
    words_size = len(word_num_map)
    to_num = lambda word:word_num_map.get(word,words_size)
    if txt_file != None:
        txt_label = get_ch_lable(txt_file)
        pass
    labels_vector = list(map(to_num,txt_label))
    return labels_vector
pass

training_data = get_ch_lable(training_file)
print("Loaded training data...")

counter = Counter(training_data)
words = sorted(counter)
print(words)
words_size = len(words)
word_num_map = dict(zip(words,range(words_size)))

print("字表大小：",words_size)
wordlabel = get_ch_lable_v(training_file,word_num_map)

#定义参数
learning_rate = 0.001
training_iters = 10000
display_step = 1000
n_input = 4

n_hidden1 = 256
n_hidden2 = 512
n_hidden3 = 512
#定义占位符
x = tf.placeholder("float",[None,n_input,1])
wordy = tf.placeholder("float",[None,words_size],name="wordy")
x1 = tf.reshape(x,[-1,n_input])
x2 = tf.split(x1,n_input,1)
#2-layer LSTM ,每层有n_hidden个utils
rnn_cell = rnn.MultiRNNCell([rnn.LSTMCell(n_hidden1),rnn.LSTMCell(n_hidden2),rnn.LSTMCell(n_hidden3)])
#通过RNN 得到输出
outputs , states = rnn.static_rnn(rnn_cell,x2,dtype = tf.float32)

#通过全连接输出指定维度
pred = tf.contrib.layers.fully_connected(outputs[-1],words_size,activation_fn = None)

#定义loss与优化器
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred , labels = wordy))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

#模型评估
correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(wordy,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

savedir = 'log/rnnword/'
saver = tf.train.Saver(max_to_keep = 1)

#启动session
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    step = 0
    offset = random.randint(0,n_input+1)
    end_offset = n_input +1 
    acc_total = 0
    loss_total = 0
    
    kpt = tf.train.latest_checkpoint(savedir)
    print("kpt:",kpt)
    startepo = 0
    if kpt != None:
        saver.restore(session,kpt)
        ind = kpt.find('-')
        startepo = int(kpt[ind+1:])
        print(startepo)
        step = startepo
        pass
    while True:#step < training_iters:
        #随机取一个位置偏移
        if offset > (len(training_data) - end_offset):
            offset = random.randint(0,n_input + 1 )
            pass
        inwords = [  [wordlabel[i]] for i in range(offset , offset+n_input)] #按照指定的位置偏移获得后 4 个文字向量，当作输入
        inwords = np.reshape(np.array(inwords) , [-1,n_input,1])
        
        out_onehot = np.zeros([words_size],dtype = float)
        out_onehot[wordlabel[offset+n_input]] = 1.0
        out_onehot = np.reshape(out_onehot,[1,-1]) #所有的字变成onehot
        
        _, acc,lossval,onehot_pred = session.run([optimizer,accuracy,loss,pred],feed_dict={x:inwords,wordy:out_onehot})
        loss_total += lossval
        acc_total += acc
        if (step +1) % display_step == 0:
            print("Iter= " +str(step +1 )+",Average Loss="+"{:.6f}".format(loss_total/display_step) + ",Average Accuracy = "+"{:.2f}%".format(100*acc_total/display_step))
            acc_total = 0
            loss_total = 0
            in2 = [words [wordlabel[i]] for i in range(offset,offset + n_input)]
            out2 = words [ wordlabel[offset+n_input]]
            out_pred = words[int(tf.argmax(onehot_pred ,1).eval())]
            print("%s - [%s] vs [%s]" % (in2,out2,out_pred))
            saver.save(session,savedir+'rnnwordtest.cpkt',global_step = step)
            pass
        step += 1
        offset += (n_input + 1) #调整下一代迭代使用的偏移量
        if 100*acc_total/display_step >=90.0:
            break
        pass
    
    print("Finished !")
    saver.save(session,savedir+"rnnwordtest.cpkt",global_step = step)
    print("Elapsed time: ",elapsed(time.time() - start_time))
    
    
    while True:
        prompt = "请输入%s个字："%n_input
        sentence = input(prompt)       
        inputword = sentence.strip()
        
        if len(inputword) != n_input:
            print("输入长度为：",len(inputword),"请输入 4 个字")
            continue
        try:
            inputword = get_ch_lable_v(None,word_num_map,inputword)
            
            for i in range(32):
                keys = np.reshape(np.array(inputword),[-1,n_input,1])
                onehot_pred = session.run(pred,feed_dict={x:keys})
                onehot_pred_index = int(tf.argmax(onehot_pred,1).eval())
                sentence = "%s%s"%(sentence,words[onehot_pred_index])
                inputword = inputword[1:]
                inputword.append(onehot_pred_index)
                pass
            print(sentence)
        except:
            print("我还没学会")












