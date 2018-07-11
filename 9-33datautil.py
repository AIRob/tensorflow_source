# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 11:50:50 2018

@author: lWX379138
"""
import sys,os,re
import random
import tensorflow as tf
import numpy as np
import collections
import jieba
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt 
jieba.load_userdict('myjiebadict.txt')

data_dir = 'cn2en/'
raw_data_dir = 'cn2en/from'
raw_data_dir_to = 'cn2en/to'
vocabulary_fileen = 'english.raw.txt'
vocabulary_filech = 'chinese.raw.txt'

plot_histograms = plot_scatter = True
vocab_size = 40000

max_num_lines = 1
max_target_size = 200
max_source_size = 200


def fenci(training_data):
    seq_list = jieba.cut(training_data)  #默认精简模式
    training_ci = ' '.join(seq_list)
    training_ci = training_ci.split()
    return training_ci
pass

#系统字符，创建字典时需要加入
_PAD = '_PAD'
_GO = '_GO'
_EOS = '_EOS'
_UNK = '_UNK'

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID =3

#文字字符替换，不属于系统字符
_NUM = '_NUM'
#Isch = true 中文，false 英文
#创建词典,max_vocabulary_size = 500代表字典里面有500个词
def create_vocabulary(vocabulary_file,raw_data_dir,max_vocabulary_size,Isch=True,normalize_digits=True):
    texts,textssz = get_ch_path_text(raw_data_dir,Isch,normalize_digits)
    print(texts[0],len(texts))
    print("行数:",len(textssz),textssz)
    
    #处理多行文本texts
    all_words = []
    for label in texts:
        print('词数：',len(label))
        all_words += [word for word in label]
        pass
    print('总词数：',len(all_words))
    
    training_label ,count,dictionary,reverse_dictionary = build_dataset(all_words,max_vocabulary_size)
    print("reverse_dictionary:",reverse_dictionary,len(reverse_dictionary))
    if not tf.gfile.Exists(vocabulary_file):
        print("Creating vocabulary %s from data %s"%(vocabulary_file,data_dir))
        if len(reverse_dictionary) > max_vocabulary_size:
            reverse_dictionary = reverse_dictionary[:max_vocabulary_size]
            pass
        with tf.gfile.GFile(vocabulary_file,mode='w') as vocab_file:
            for w in reverse_dictionary:
                print(reverse_dictionary[w])
                vocab_file.write(reverse_dictionary[w] +'\n')
                pass
            pass
        pass
    else:
        print("already have vocabulary ! do nothing !!!!!")
        pass
    return training_label,count,dictionary,reverse_dictionary,textssz
pass

def build_dataset(words,n_words):
    #Process raw inputs into a dataset
    count = [[_PAD,-1],[_GO,-1],[_EOS,-1],[_UNK,-1]]
    count.extend(collections.Counter(words).most_common(n_words - 1) )
    dictionary = dict()
    for word,_ in count:
        dictionary[word] = len(dictionary)
        pass
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0   #dictionary['UNK']
            unk_count += 1
            pass
        data.append(index)
        pass
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(),dictionary.keys()))
    return data,count,dictionary,reversed_dictionary
pass
    

def main():
    vocabulary_filenameen = os.path.join(data_dir,vocabulary_fileen)
    vocabulary_filenamech = os.path.join(data_dir,vocabulary_filech)
    #######################################################################################
    #创建英文字典
    training_dataen,counten,dictionaryen,reverse_dictionaryen,textsszen = create_vocabulary(vocabulary_filenameen,raw_data_dir,vocab_size,Isch = False,normalize_digits = True)
    print("training_data",len(training_dataen))
    print("dictionary",len(dictionaryen))
    
    #创建中文字典
    training_datach,countch,dictionarych,reverse_dictionarych,textsszch = create_vocabulary(vocabulary_filenamech,raw_data_dir_to,vocab_size,Isch = True,normalize_digits = True)
    print("training_datach",len(training_datach))
    print("dictionarych",len(dictionarych))   
    
    vocaben,rev_vocaben = initialize_vocabulary(vocabulary_filenameen)
    vocabch,rev_vocabch = initialize_vocabulary(vocabulary_filenamech)
    
    print(len(rev_vocaben))
    textdir_to_idsdir(raw_data_dir,data_dir+'fromids/',vocaben,normalize_digits=True,Isch =False)
    textdir_to_idsdir(raw_data_dir,data_dir+'toids/',  vocabch,normalize_digits=True,Isch =True)
    
    #分析样本分布
    filesfrom , _ = getRawFileList(data_dir+'fromids/')
    filesto ,_ = getRawFileList(data_dir + 'toids/')
    source_train_file_path = filesfrom[0]
    target_train_file_path = filesto[0]
    analysisfile(source_train_file_path,target_train_file_path)
    
    pass
pass

def initialize_vocabulary(vocabulary_path):
    if tf.gfile.Exists(vocabulary_path):
        rev_vocab = []
        with tf.gfile.GFile(vocabulary_path,mode='r') as f:
            rev_vocab.extend(f.readlines())
            pass
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x,y) for (y,x) in enumerate(rev_vocab)])
        return vocab,rev_vocab
    else:
        raise ValueError('Vocabulary file %s not found.',vocabulary_path)
    pass
pass

#将文件批量转成ids文件
def textdir_to_idsdir(textdir,idsdir,vocab,normalize_digits = True,Isch = True):
    text_files , filenames = getRawFileList(textdir)
    
    if len(text_files) == 0:
        raise ValueError('Err:no files in ',raw_data_dir)
    
    print(len(text_files),'files one is',text_files[0])
    
    for text_file , name in zip(text_files,filenames):
        print(text_file,idsdir+name)
        textfile_to_idsfile(text_file,idsdir+name,vocab,normalize_digits,Isch)
        pass
    pass
pass

#获取文件列表
def getRawFileList(path):
    files = []
    names = []
    for f in os.listdir(path):
        if not f.endswith('~') or not f =='':
            files.append(os.path.join(path,f))
            names.append(f)
            pass
        pass
    return files,names
pass

#读取分词后的中文词
def get_ch_lable(txt_file,Isch = True,normalize_digits = False):
    labels = list() 
    labelssz = []
    with open(txt_file,'rb') as f:
        for label in f:
            linstr1 = label.decode('utf-8')
            if normalize_digits:
                linstr1 = re.sub('\d+',_NUM,linstr1)
                pass
            notoken = basic_tokenizer(linstr1)
            if Isch:
                notoken = fenci(notoken)
            else:
                notoken = notoken.split()
                pass
            
            labels.extend(notoken)
            labelssz.append(len(labels))
            pass
        pass
    return labels,labelssz
pass

#获取文件中文本
def get_ch_path_text(raw_data_dir,Isch = True,normalize_digits=False):
    text_files , _ = getRawFileList(raw_data_dir)
    labels = []
    
    training_dataszs = list([0])
    
    if len(text_files) == 0:
        print("err:no files in ",raw_data_dir)
        return labels
    print(len(text_files),'files ,one is',text_files[0])
    random.shuffle(text_files)
    
    for text_file in text_files:
        training_data , training_datasz = get_ch_lable(text_file,Isch,normalize_digits)
        
        training_ci = np.array(training_data)
        training_ci = np.reshape(training_ci,[-1,])
        labels.append(training_ci)
        
        training_datasz = np.array(training_datasz) + training_dataszs[-1]
        training_dataszs.extend(list(training_datasz))
        print("here",training_dataszs)
        pass
    return labels,training_dataszs
pass

def basic_tokenizer(sentence):
    _WORD_SPLIT = "([.,!?\"':;)(])"
    _CHWORD_SPLIT = '、|。|，|‘|’'
    str1 = ''
    for i in re.split(_CHWORD_SPLIT,sentence):
        str1 = str1 + i
        pass
    str2 = ''
    for i in re.split(_WORD_SPLIT,str1):
        str2 = str2 + i
        pass
    return str2
pass

#将句子转成索引 ids
def sentence_to_ids(sentence,vocabulary,normalize_digits = True,Isch = True):
    if normalize_digits:
        sentence = re.sub('\d+',_NUM,sentence)
        pass
    notoken = basic_tokenizer(sentence)
    if Isch:
        notoken = fenci(notoken)
    else:
        notoken = notoken.split()
        pass
    
    idsdata = [vocabulary.get(w,UNK_ID) for w in notoken]
    return idsdata
pass

#将文件中的内容转成ids，不是windows下的文件需要使用utf-8 编码格式
def textfile_to_idsfile(data_file_name,target_file_name,vocab,normalize_digtis=True,Isch=True):
    if not tf.gfile.Exists(target_file_name):
        print('Tokenizing data in %s'%data_file_name)
        with tf.gfile.GFile(data_file_name,mode='rb') as data_file:
            with tf.gfile.GFile(target_file_name,mode='w') as ids_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 100000 == 0:
                        print("tokenizing line %d" % counter)
                        pass
                    token_ids = sentence_to_ids(line.decode('utf8'),vocab,normalize_digtis,Isch)
                    ids_file.write(" ".join([str(tok) for tok in token_ids]) + '\n')
                    pass
                pass
            pass
        pass
    pass
pass

def ids2texts(indices,rev_vocab):
    texts = []
    for index in indices:
        texts.append(rev_vocab[index])
        pass
    return texts
pass

#分析文本
def analysisfile(source_file,target_file):
    source_lengths = []
    target_lengths = []
    
    with tf.gfile.GFile(source_file,mode='r') as s_file:
        with tf.gfile.GFile(target_file,mode='r') as t_file:
            source = s_file.readline()
            target = t_file.readline()
            counter = 0
            
            while source and target:
                counter += 1
                if counter % 100000 == 0:
                    print("reading data line %d" % counter)
                    sys.stdout.flush()
                    pass
                num_source_ids = len(source.split())
                source_lengths.append(num_source_ids)
                num_target_ids = len(target.split()) + 1 #加1 是EOS
                target_lengths.append(num_target_ids)
                source,target = s_file.readline(),t_file.readline()
                pass
            pass
        pass
    
    print(target_lengths,source_lengths)
    if plot_histograms:
        plot_histo_lengths("target lengths ",target_lengths)
        plot_histo_lengths("source lengths",source_lengths)
    if plot_scatter:
        plot_scatter_lengths("target vs source length","source length","target length",source_lengths,target_lengths)
        pass
    pass
pass

def plot_scatter_lengths(title,x_title,y_title,x_lengths,y_lengths):
    plt.scatter(x_lengths,y_lengths)
    plt.title(title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.ylim(0,max(y_lengths))
    plt.xlim(0,max(x_lengths))
    plt.show()
    pass
pass

def plot_histo_lengths(title,lengths):
    mu = np.std(lengths)
    sigma = np.mean(lengths)
    x = np.array(lengths)
    n,bins,patches = plt.hist(x,50,facecolor='green',alpha = 0.5)
    y = mlab.normpdf(bins,mu,sigma)
    plt.plot(bins,y,'r--')
    plt.title(title)
    plt.xlabel('Length')
    plt.ylabel("Number of sequences")
    plt.xlim(0,max(lengths))
    plt.show()
    pass
pass
    
if __name__ =='__main__':   
    main()

