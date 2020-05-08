#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 13:28:07 2019

@author: hb
"""
import torch
import torch.utils.data as D
import pickle
import codecs
import numpy as np
embed_dim=768
from pytorch_pretrained_bert import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('.../bert-base-uncased')

count=0
with codecs.open('.../bert-base-uncased/vocab.txt','r') as inp:
    word2id={}
    id2word={}
    lines=inp.readlines()
    for line in lines:
        line=line.strip()
        word2id[line]=count
        id2word[count]=line
        count+=1
    inp.close()

def tokenize(text):
    words=[]
    words.append('[CLS]')
    for word in text.split():
        words.append(word)
    words.append('[SEP]')
    return words    

def get_ids(text,word2id):
    ids=[]
    for word in text:
        if word in word2id:
            ids.append(word2id[word])
            
        else:
            ids.append(word2id['[UNK]'])
    return ids
#===========================================================================
with codecs.open('.../datasets/semeval14/Restaurants_Train.xml.seg','r','utf-8') as inp:
        lines=inp.readlines()
        inp.close()
        max_seq_len=0
        max_aspect_len=0
        max_left_len=0
        text_all=[]
        aspect=[]
        polarity=[]
        text_left_all=[]
        #定义存放长度的列表
        for i in range(0,len(lines),3):
            polarity_i=[]
            #评论信息
            text_left,_,text_right=[s.lower()  for s in lines[i].partition('$T$')]
            text_left=text_left.strip()
            text_right=text_right.strip()
            text_string=text_left+' '+lines[i+1].lower().strip()+' '+text_right
            text_string=tokenize(text_string)
            token_text=get_ids(text_string,word2id)
#            print(token_text)
            seq_leni=len(token_text)
            if max_seq_len<seq_leni:
               max_seq_len=seq_leni            
            #left信息

            text_left=tokenize(text_left) 
            token_left=get_ids(text_left,word2id)            
            left_leni=len(token_left)    
            if max_left_len<left_leni:
               max_left_len=left_leni            

            #aspect信息
#            print('sss',lines[i+1].lower().strip())
            aspect_i=tokenize(lines[i+1].lower().strip()) 
#            print('aspect_i',aspect_i)
            token_aspect=get_ids(aspect_i,word2id)  
            aspect_leni=len(token_aspect)
            if max_aspect_len<aspect_leni:
                max_aspect_len=aspect_leni
            
            #情感极性
            polarity_i.append(int(lines[i+2].strip())+1)
#            print('polarity_i',polarity_i)
            aspect.append(token_aspect)
            polarity.extend(polarity_i)
            text_left_all.append(token_left)
            text_all.append(token_text)

def get_len(texts):
    text_len=[]
    for text in texts:
        text_len.append(len(text))
    return text_len
#通过这个函数要得到text_all_indices,aspect_indices,text_left_indices其中第一个需要做padding，后面两个不用
def sequence_padding(text_all,max_seq_len):
    ids=[]
    for text in text_all:      
#        if len(text)>max_seq_len:
#            text=text[:max_seq_len]
        text.extend([word2id['[PAD]']]*(max_seq_len-len(text)))
        ids.append(text)
    return ids
#============================================================================
text_len=get_len(text_all)
aspect_len=get_len(aspect)
#print(aspect_len)
left_len=get_len(text_left_all)
text_ids=sequence_padding(text_all,max_seq_len)
aspect_ids=sequence_padding(aspect,max_aspect_len)
text_left_ids=sequence_padding(text_left_all,max_left_len)
#将数据处理得到的下游任务需要的部分保存
with open('train.pkl','wb') as outp:
    pickle.dump(polarity,outp)
    pickle.dump(text_ids,outp)
    pickle.dump(aspect_ids,outp)
    pickle.dump(text_left_ids,outp)
    pickle.dump(text_len,outp)
    pickle.dump(aspect_len,outp)
    pickle.dump(left_len,outp)

with codecs.open('.../datasets/semeval14/Restaurants_Test_Gold.xml.seg','r','utf-8') as inp:
        lines=inp.readlines()
#        lines=lines.rstrip('\n')
        inp.close()
        text_all_t=[]
        aspect_t=[]
        polarity_t=[]
        text_left_all_t=[]
        for i in range(0,len(lines),3):
            polarity_i=[]
            #评论信息
            text_left,_,text_right=[s.lower()  for s in lines[i].partition('$T$')]
            text_left=text_left.strip()
            text_right=text_right.strip()
            text_string=text_left+' '+lines[i+1].lower().strip()+' '+text_right
            text_string=tokenize(text_string) 
           
            token_text=get_ids(text_string,word2id)
            seq_leni=len(token_text)
            if max_seq_len<seq_leni:
               max_seq_len=seq_leni            
            #left信息
            text_left=tokenize(text_left)         
            token_left=get_ids(text_left,word2id)            
            left_leni=len(token_left)  
            if max_left_len<left_leni:
               max_left_len=left_leni            
            #aspect信息
            aspect_i=tokenize(lines[i+1].lower().strip())           
            token_aspect=get_ids(aspect_i,word2id)  
            aspect_leni=len(aspect_i)
            if max_aspect_len<aspect_leni:
                max_aspect_len=aspect_leni
            
            #情感极性
            polarity_i.append(int(lines[i+2].strip())+1)
            aspect_t.append(token_aspect)
            polarity_t.extend(polarity_i)
            text_left_all_t.append(token_left)
            text_all_t.append(token_text)
text_len_t=get_len(text_all_t)
aspect_len_t=get_len(aspect_t)
left_len_t=get_len(text_left_all_t)
text_ids_t=sequence_padding(text_all_t,max_seq_len)
aspect_ids_t=sequence_padding(aspect_t,max_aspect_len)
text_left_ids_t=sequence_padding(text_left_all_t,max_left_len)
with open('test.pkl','wb') as outp:
    pickle.dump(polarity_t,outp)
    pickle.dump(text_ids_t,outp)
    pickle.dump(aspect_ids_t,outp)
    pickle.dump(text_left_ids_t,outp)
    pickle.dump(text_len_t,outp)
    pickle.dump(aspect_len_t,outp)
    pickle.dump(left_len_t,outp)



















