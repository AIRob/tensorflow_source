# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 14:26:17 2018

@author: lWX379138
带有注意力机制的 sequence-to-sequence 模型
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange    #pylint : disable = redefined-builtin
import tensorflow as tf
data_utils = __import__('9-33datautil')
#import datautil as data_utils

class Seq2SeqModel(object):
    """
    带有注意力机制并且具有multiple buckets 的 Sequence-to-Sequence 模型
    这个类实现一个多层循环网络组成的编码器和一个具体注意力机制的编码器，完全
    是按照论文：
    http://arxiv.org/abs/1412.7449 - 中所描述的机制实现.更多细节可以参照
    论文内容
    这个class 除了使用LSTM cells 还可以使用GRU cells，还使用 sampled 
    softmax 来处理 大量词汇的输出。
    在论文 http://arxiv.org/abs/1412.2007 中第三节描述了sampled softmax。
    在论文 http://arxiv.org/abs/1409.0473 里面还有一个关于这个模型的一个单层的
    使用双向RNN 编码器版本.
    """
    def __init__ (self,
                  source_vocab_size,
                  target_vocab_size,
                  buckets,
                  size,
                  num_layers,
                  dropout_keep_prob,
                  max_gradient_norm,
                  batch_size,
                  learning_rate,
                  learning_rate_decay_factor,
                  use_lstm = False,
                  num_samples = 512,
                  forward_only = False,
                  dtype = tf.float32):
        '''创建模型
        Args:
            source_vocab_size : 原词汇的大小
            target_vocab_size :目标词汇的大小
            buckets :一个（I,O）的list ，I代表输入的最大长度,O代表输出的最大长度,例如 [(2,4),(8,16)]
            size:模型中每层utils 个数
            num_layers: 模型的层数
            max_gradient_norm:截断梯度的阀值
            batch_size:训练中的批次数据大小
            learning_rate: 开始学习率
            learning_rate_decay_factor: 退化学习率的衰减参数
            ust_lstm:如果是True , 使用 LSTM cells 替代GRU cells
            num_samples : sampled softmax 的样本个数
            forward_only：如果设置了，模型只有正向传播
            dtype：internal variables 的类型
            
        '''
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.buckets = buckets
        self.batch_size = batch_size
        self.dropout_keep_prob_output = dropout_keep_prob
        self.dropout_keep_prob_input = dropout_keep_prob
        self.learning_rate = tf.Variable(float(learning_rate),trainable=False,dtype=dtype)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0,trainable= False)
        
        #如果使用 samplesoftmax ,需要一个输出的映射
        output_projection = None
        softmax_loss_function = None 
        #当采样率小于 vocabulary size 时 smapled softmax 才有意义
        if num_samples > 0 and num_samples < self.target_vocab_size:
            w_t = tf.get_variable("proj_w",[self.target_vocab_size,size],dtype=dtype)
            w = tf.transpose(w_t)
            b = tf.get_variable("proj_b",[self.target_vocab_size],dtype =dtype)
            output_projection = (w,b)
            pass
        pass
    
    def sampled_loss(labels,logits):
        labels = tf.reshape(labels , [-1,1])
        #需要使用 32 bit 的浮点类型来计算 sampled_softmax_loss 才能 避免数值的不稳定性、
        local_w_t = tf.cast(w_t,tf.float32)
        local_b = tf.cast(b,tf.float32)
        local_inputs = tf.cast(logits,tf.float32)
        return tf.cast(tf.nn.sampled_softmax_loss(weights= local_w_t,
                                                  biases = local_b,
                                                  labels=labels,
                                                  inputs=local_inputs,
                                                  num_sampled=num_samples,
                                                  num_classes=self.target_vocab_size),
                       dtype)
    
    softmax_loss_function = sampled_loss
    
    #使用词嵌入量 embedding作为输入
    def seq2seq_f(encoder_inputs,decoder_inputs,do_decode):
        with tf.variable_scope("GRU") as scope:
            cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(size),
                                                 input_keep_prob=self.dropout_keep_prob_input,
                                                 output_keep_prob=self.dropout_keep_prob_output)
            if num_layers > 1:
                cell = tf.contrib.rnn.MultiRNNCell( [cell] * num_layers)
                pass
            pass
        
        print('new a cell')
        return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(encoder_inputs,
                                                                     decoder_inputs,
                                                                     cell,
                                                                     num_encoder_symbols=source_vocab_size,
                                                                     num_decoder_symbols=target_vocab_size,
                                                                     embedding_size=size,
                                                                     output_projection=output_projection,
                                                                     feed_previous = do_decode,
                                                                     dtype=dtype)
    def model_with_buckets():
        #注入数据
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        for i in xrange(buckets[-1][0]):#最后的bucket 是最大的
            self.encoder_inputs.append(tf.placeholder(tf.int32,shape=[None],name="encoder{0}".format(i)))
            pass
        for i in xrange(buckets[-1][1]+1):
            self.decoder_inputs.append(tf.placeholder(tf.int32,shape=[None],name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(dtype,shape=[None],name="weight{0}".format(i)))
            pass
        #将解码器移动一位得到targets
        targets = [self.decoder_inputs[i+1] for i in xrange(len(self.decoder_inputs) - 1)]    
        #许梿的输出和loss定义
        if forward_only:
            self.outputs,self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(self.encoder_inputs,
                                                                                    self.decoder_inputs,
                                                                                    targets,
                                                                                    self.target_weights,
                                                                                    buckets,
                                                                                    lambda x ,y :seq2seq_f(x,y,True),
                                                                                    softmax_loss_function= softmax_loss_function)
            #如果使用了输出映射,需要为解码器映射输出处理
            if output_projection is not None:
                for b in xrange(len(buckets)):
                    self.outputs[b] = [ tf.matmul(output,output_projection[0] ) + output_projection[1] for outout in self.outputs[b]  ]
                    pass
                pass
            pass
        else:
            self.outputs,self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(self.encoder_inputs,
                                                                                    self.decoder_inputs,
                                                                                    targets,
                                                                                    self.target_weights,
                                                                                    buckets,
                                                                                    lambda x,y : seq2seq_f(x,y,False),
                                                                                    softmax_loss_function = softmax_loss_function )
            pass
        
        #梯度下降更新操作
        params = tf.trainable_variables()
        if not forward_only:
            self.gradient_norms = []
            self.updates = []
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            for b in xrange(len(buckets)):
                gradients = tf.gradients(self.losses[b],params)
                clipped_gradients , norm = tf.clip_by_global_norm(gradients,max_gradient_norm)
                self.gradient_norms.append(norm)
                self.updates.append(opt.apply_gradients(zip(clipped_gradients,params),
                                                        global_step=self.global_step))
                pass
            pass
        self.saver = tf.train.Saver(tf.global_variables())
    
    def get_batch(self,data,bucket_id):
        '''
        在迭代过程中，从指定bucket中获取一个随机批次数据
        Args:
            data:一个大小为len(self.buckets) 的tuple,包含创建一个batch中的输入输出的lists.
            bucket_id:方便以后调用的triple (encoder_inputs,decoder_inputs,target_weights).
        '''
        encoder_size ,decoder_size = self.buckets[bucket_id]
        encoder_inputs ,decoder_inputs = [] , []
        
        #获取一个随机批次的数据作为编码器和解码器的输入
        #如果需要时会有pad操作，同时反转encoder的输入顺序，并且decoder添加GO
        for _ in xrange(self.batch_size):
            encoder_input , decoder_input = random.choice(data[bucket_id])
            
            #pad和反转Encoder 的输入数据
            encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))
            
            #为Decoder 输入数据添加一个额外的“GO”，并且进行pad
            decoder_pad_size = decoder_size - len(decoder_input) -1
            decoder_inputs.append([data_utils.GO_ID] + decoder_input + [data_utils.PAD_ID] * decoder_pad_size)
            pass
        #从上面选择好的数据中创建batch-major vectors
        batch_encoder_inputs , batch_decoder_inputs ,batch_weights = [],[],[]
        
        for length_idx in xrange(encoder_size):
            batch_encoder_inputs.append(np.array([encoder_inputs[batch_idx][length_idx] for batch_idx in xrange(self.batch_size)],dtype=np.int32 ))
            pass
        
        for length_idx in xrange(decoder_size):
            batch_decoder_inputs.append(np.array([decoder_inputs[batch_idx][length_idx] for batch_idx in xrange(self.batch_size) ] ,dtype=np.int32))
            
            #定义target_weights 变量,默认是 1 ，如果对应的targets 是 padding,则target_weights 就为 0
            batch_weight = np.ones(self.batch_size,dtype=np.float32)
            for batch_idx in xrange(self.batch_size):
                #如果对应的输出target 是一个PAD符号，就将weight设为 0 
                #将decoder_input 向前移动一位得到 对应的target
                if length_idx < decoder_size - 1 :
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == decoder_size -1 or target == data_utils.PAD_ID :
                    batch_weight[batch_idx] - 0.0
                    pass
                pass
            batch_weights.append(batch_weight)
            pass
        return batch_encoder_inputs,batch_decoder_inputs,batch_weights
    
    def step(self,session,encoder_inputs,decoder_inputs,target_weights,bucket_id,forward_only):
        '''
        注入给定输入数据步骤
        Args:
            session:tensorflow 所用的session
            encoder_inputs : 用来注入encoder输入数据的numpy int vectors 类型的lists
            decoder_inputs : 用来注入decoder输入数据的numpy int vectors 类型的lists
            target_weights ： 用来注入target weights的numpy int vectors 类型的lists
            bucket_id：which bucket of the model to use
            forward_only：只进行正向传播
        returns:
            一个由 gradient norm (不做反向时为None)，average perplexity ,and the outputs 组成的 triple
        Raises：
            ValueError：如果encoder_inputs,decoder_inputs,或者target）weights 长度与指定的bucket_id 的bucket size 不符合
        '''
        #检查长度
        encoder_size ,decoder_size = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket,%d != %d"%(len(encoder_inputs),decoder_size))
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket,%d != %d"%(len(decoder_inputs),decoder_size))   
        if len(target_weights) != decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket,%d != %d"%(len(target_weights),decoder_size))  
        
        #定义 Input feed
        input_feed = {}
        for l in xrange(encoder_size):
            input_feed[self.encoder_input[l].name] = encoder_inputs[l]
        for l in xrange(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]
            pass
        
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size],dtype = np.int32)
        
        #定义output feed
        if not forward_only:
            output_feed = [self.updates[bucket_id],
                           self.gradients_norms[bucket_id],
                           self.losses[bucket_id]]
        else:
            output_feed = [self.losses[bucket_id]]
            for l in xrange(decoder_size):
                output_feed.append(self.outputs[bucket_id])
                pass
            pass
        
        outputs = session.run(output_feed,input_feed)
        if not forward_only:
            return outputs[1],outputs[2],None
        else:
            return None,outputs[0],outputs[1:]
        pass
    
    
        