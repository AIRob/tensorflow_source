# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 17:27:31 2018

@author: lWX379138
"""
import tensorflow as tf
import numpy as np
import os
import codecs  #读写编码问题

from python_speech_features import mfcc    #使用pip install 额外安装
import scipy.io.wavfile as wav



#读取wav文件对应的label
def get_wavs_lables(wav_path ,label_file):
    #获取训练用的WAV文件路径列表
    wav_files = []
    for (dirpath,dirnames,filenames ) in os.walk(wav_path):        
        for filename in filenames:
            if filename.endswith('.wav') or filename.endswith('.WAV'):
                filename_path = os.sep.join([dirpath,filename])
                if os.stat(filename_path).st_size < 240000:#剔除掉一些小文件
                    continue
                wav_files.append(filename_path.replace('\\','/'))#统一用正斜杠
                pass
            pass
        pass
    labels_dict = {}
    f = codecs.open(label_file,'r','utf-8')
    for label in f:
        sp_list = label.split('||')
        if len(sp_list) >= 2:#排除多余的回车符
            #print(sp_list,len(sp_list),type(sp_list))
            label_id = sp_list[0]
            label_text =sp_list[1].replace('\n','')
            #print('label_id:',label_id)
            #print(label_id,label_id.decode('utf-8'))
            #print(label_text,label_text.decode('utf-8'))
            labels_dict[label_id] = label_text
            pass
        pass
    pass
    labels = []
    new_wav_files = []
    for wav_file in wav_files:
        wav_id = wav_file.replace('\\','/')#os.path.basename(wav_file).split('.')[0]
        #print('wav_id:',wav_id)
        if wav_id in labels_dict.keys():##?
            labels.append(labels_dict[wav_id])
            new_wav_files.append(wav_file)
            pass
        pass
    return new_wav_files,labels
pass    

'''
遍历所有音频文件及文件
将音频文件 调用 audiofile_to_input_vector 转成MFCC
调用get_ch_lable_v 将文本转成向量
'''
def get_audio_and_transcriptch(txt_files,wav_files,n_input,n_context,word_num_map,txt_labels = None):
    
    audio = []
    audio_len =  []
    transcript = []
    transcript_len = []
    if txt_files != None:
        txt_labels = txt_files
        pass
    #print(type(txt_labels),len(txt_labels))
    for txt_obj , wav_file in zip(txt_labels,wav_files):
        #载入音频数据并转化为特征值
        audio_data = audiofile_to_input_vector(wav_file,n_input,n_context)
        audio_data = audio_data.astype('float32')
        
        audio.append(audio_data)
        audio_len.append(np.int32(len(audio_data)))
        
        #载入音频对应的文本
        target = []
        if txt_files != None: #txt_obj 是文件
            target = get_ch_lable_v(txt_obj,word_num_map)
        else:
            target = get_ch_lable_v(None,word_num_map,txt_obj)  #txt_obj 是 labels
            pass
        transcript.append(target)
        transcript_len.append(len(target))
        pass
    
    audio = np.asarray(audio)
    audio_len = np.asarray(audio_len)
    transcript = np.asarray(transcript)
    transcript_len = np.asarray(transcript_len)
    
    return audio,audio_len,transcript,transcript_len
pass

'''
audiofile_to_input_vector 转成MFCC
'''
def audiofile_to_input_vector(audio_filename,numcep,numcontext):
    #加载wav 文件
    fs,audio = wav.read(audio_filename)
    
    #获取mfcc cofficients
    orig_inputs = mfcc(audio,samplerate = fs , numcep = numcep)
    #print(orig_inputs.shape)
    orig_inputs = orig_inputs[::2]              #（139,26）
    #print(orig_inputs.shape)
    
    train_inputs = np.array([],np.float32)
    train_inputs.resize((orig_inputs.shape[0],numcep + 2*numcep*numcontext))

    empty_mfcc = np.array([])
    empty_mfcc.resize((numcep)) 
    
    #准备输入数据。输入数据的格式由三部分安装顺序拼接而成，分成当前样本的前9个序列样本，当前样本序列、后9个序列
    #print(train_inputs.shape[0])
    time_slices = range(train_inputs.shape[0])  # 139 切片
    context_past_min = time_slices[0] + numcontext
    context_future_max = time_slices[-1] - numcontext  #[9,1,2,...,137,129]
    for time_slice in time_slices :
        #前9个补0，mfcc features
        need_empty_past = max(0,(context_past_min - time_slice))
        empty_source_past = list(empty_mfcc for empty_slots in range(need_empty_past))
        data_source_past = orig_inputs[max(0,(time_slice - numcontext)):time_slice]
        assert(len(empty_source_past) + len(data_source_past) == numcontext )
        
        #后9个补0,mfcc features
        need_empty_future  = max(0,(time_slice - context_future_max))
        empty_source_future = list(empty_mfcc for empty_slots in range(need_empty_future))
        data_source_future = orig_inputs[time_slice + 1 : time_slice+numcontext +1 ]
        assert(len(empty_source_future) + len(data_source_future) == numcontext)
        
        if need_empty_past :
            past = np.concatenate((empty_source_past,data_source_past))
        else:
            past = data_source_past
        
        if need_empty_future:
            future = np.concatenate((data_source_future,empty_source_future))
        else:
            future = data_source_future
            
        past = np.reshape(past,numcontext*numcep)
        now = orig_inputs[time_slice]
        future = np.reshape(future,numcontext*numcep)
        
        train_inputs[time_slice] = np.concatenate((past,now,future))
        assert(len(train_inputs[time_slice]) == numcep + 2 * numcep * numcontext)
        

        pass
    #将数据使用正太分布标准化，减去均值然后再除以方差
    train_inputs = (train_inputs - np.mean(train_inputs)) / np.std(train_inputs)
    
    return train_inputs    
pass

'''
批次音频数据对齐
可以支持 补0 和 截断 两个操作
post 代表：后补0
pre  代表; 截断
'''
def pad_sequences(sequences,maxlen=None,dtype = np.float32,padding = 'post',truncating='post',value=0.):
    
    lengths = np.asarray([len(s) for s in sequences ] , dtype=np.int64)
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
        
    #从第一个非空的序列得到样本形状
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break
        pass
    
    x = (np.ones((nb_samples,maxlen) + sample_shape) * value).astype(dtype)
    for idx ,s in enumerate(sequences):
        if len(s) == 0:
            continue
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating =='post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)
        
        #检查trunc
        trunc = np.asarray(trunc,dtype = dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('shape of sample %s of sequence at position %s is different from expected shape %s'
                                %(trunc.shape[1:],idx,sample_shape))
            pass
        
        if padding =='post':
            x[idx, : len(trunc)] = trunc
        elif padding =='pre':
            x[idx,-len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood'%padding)
            pass
        pass
    
    return x , lengths
    
pass

'''
文本样本的转化
将文字转化成具体的向量
'''
def get_ch_lable_v(txt_file,word_num_map,txt_label = None):
    
    words_size = len(word_num_map)
    
    to_num = lambda word:word_num_map.get(word,words_size)
    
    if txt_file != None:
        txt_label =get_ch_lable(txt_file)
        pass
    
    labels_vector = list(map(to_num,txt_label))
    
    return labels_vector
pass

def get_ch_lable(txt_file):
    labels = ''
    with open(txt_file,'rb') as f:
        for label in f:
            labels = labels + label.decode('utf-8')
            #labels = labels + label.decode('gb2312')
            pass
        pass
    
    return labels
pass

'''
密集矩阵转成稀疏矩阵
'''
def sparse_tuple_from(sequences,dtype=np.int32):
    
    indices = []
    values = []
    
    for n ,seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq) , range(len(seq))))
        values.extend(seq)
        pass
    
    indices = np.asarray(indices,dtype = np.int64)
    values = np.asarray(values,dtype = dtype)
    shape = np.asarray([len(sequences),indices.max(0)[1] + 1],dtype=np.int64)
    
    return indices , values , shape
pass

#常量
SPACE_TOKEN = '<space>'     # space 符号
SPACE_TNDEX = 0             # 0 为 space 索引
FIRST_INDEX = ord('a') -1 

def sparse_tuple_to_texts_ch(tuple,words):
    indices = tuple[0]
    values = tuple[1]
    results = [''] * tuple[2][0]
    for i in range(len(indices)):
        index = indices[i][0]
        c = values[i]
        
        c = ' ' if c == SPACE_TNDEX else words[c]
        results[index] = results[index] + c
    #返回strings 的List
    return results
pass


def ndarray_to_text_ch(value,words):
    results = ''
    for i in range(len(value)):
        results += words[value[i]]
        pass
    return results.replace('´',' ')
pass
    



