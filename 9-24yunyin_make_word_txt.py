# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 11:19:25 2018

@author: lWX379138
用于从.wav 和 .wav.trn 文件生成word.txt
"""
import sys,os
import codecs

#f = codecs.open('c:/intimate.txt','a','utf-8')
#f.write(s.decode('gbk'))

def wavs2wordText(wav_path,word_fn):
    if os.path.exists(word_fn):
        os.remove(word_fn)
    file_lists = os.listdir(wav_path)
    lines = ''
    for fn in file_lists:
        if fn.endswith('.wav') or fn.endswith('.WAV'):
            text_fn = os.path.join(wav_path,fn+'.trn').replace('\\','/')
            if os.path.exists(text_fn):
                f = open(text_fn,'rb')
                all_lines = f.readlines()
                for line in all_lines:
                    line = (line).decode('utf-8').replace('\n','')#.replace(' ','')
                    lines += os.path.join(wav_path,fn)+'||'+ line+'\n' 
                    break
                f.close()
                pass
            pass
        pass
    f_w = codecs.open(word_fn,'a','utf-8')
    f_w.write(lines)     
    f_w.close()
    pass
pass



if __name__ =='__main__':
    
    wav_path = '../thchs30/data_thchs30/data/'
    word_fn = '../thchs30/data_thchs30/doc/train.word.txt'    
    
    wavs2wordText(wav_path,word_fn)
    
    print('end')
    pass
pass
    


