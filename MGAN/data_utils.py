#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 13:28:07 2019

@author: hb
"""
import torch
import torch.utils.data as D
import pickle
import string
import codecs
import numpy as np

embed_dim=200

from pytorch_pretrained_bert import BertTokenizer


#去除文本中的标点符号，得到文本中的一个个单词
def get_word(text):
    for i in text:
            if i in string.punctuation:
                text=text.replace(i," ")
    text=text.split()
    return text
    

def get_word2id(text_all):
        word2id={}
        id2word={}
        idx=1
        word2id['BLANK']=0
        id2word[0]='BLANK'
        for text in text_all:
                for word in text:
                    if word not in word2id:
                        word2id[word]=idx
                        id2word[idx]=word
                        idx+=1
        word2id['UNKNOW']=len(word2id)+1
        id2word[len(word2id)+1]='UNKNOW'

        return word2id,id2word
#print(get_word2id(text_all))
#===========================================================================


with codecs.open('.../datasets/semeval14/Restaurants_Train.xml.seg','r','utf-8') as inp:
        lines=inp.readlines()
#        lines=lines.rstrip('\n')
        inp.close()
        max_seq_len=0
        max_aspect_len=0
        max_left_len=0
        text_all=[]
        aspect=[]
        polarity=[]
        text_left_all=[]
        for i in range(0,len(lines),3):
            polarity_i=[]
            #评论信息
            text_left,_,text_right=[s.lower()  for s in lines[i].partition('$T$')]
            text_string=text_left+lines[i+1].lower().strip()+text_right
            text_string=get_word(text_string)            
            
            text_left=get_word(text_left)
            
            seq_leni=len(text_string)

            if max_seq_len<seq_leni:
               max_seq_len=seq_leni
            #aspect信息
            aspect_i=get_word(lines[i+1].lower().strip())
            aspect_leni=len(aspect_i)
            if max_aspect_len<aspect_leni:
                max_aspect_len=aspect_leni
            
            left_leni=len(text_left)    
            if max_left_len<left_leni:
               max_left_len=left_leni
            #情感极性
            polarity_i.append(int(lines[i+2].strip())+1)
            aspect.append(aspect_i)
            polarity.extend(polarity_i)
            text_left_all.append(text_left)
            text_all.append(text_string)

#通过这个函数要得到text_all_indices,aspect_indices,text_left_indices其中第一个需要做padding，后面两个不用
def sequence_padding(text_all,word2id,max_seq_len):
    ids=[]
    for text in text_all:
        idx=[]
        for word in text:
            if word in word2id:
                idx.append(word2id[word])
            else:
                idx.append(word2id['UNKNOW'])
        if len(idx)>max_seq_len:
            idx=idx[:max_seq_len]
        idx.extend([word2id['BLANK']]*(max_seq_len-len(idx)))
        ids.append(idx) 
    return ids

#============================================================================
#下面两个函数是为了得到embedding_matrix        
word2vec={}
with codecs.open('.../glove.txt','r','utf-8') as inp:
    lines=inp.readlines()
    for line in lines:
        word=line.split()[0]
        vector=line.split()[1:]
        vector=list(map(float,vector))
        word2vec[word]=vector
#    inp.close()
    word2vec['UNKNOW']=[1]*200
    word2vec['BLANK']=[0]*200


def embedding_matrix(word2id,word2vec):
    embedding_matrix=np.zeros((len(word2id)+1,embed_dim))
    for word,i in word2id.items():
        if word in word2vec:
            vec=word2vec[word]
        else:
            vec=word2vec['UNKNOW']
            
        embedding_matrix[i]=vec

    return embedding_matrix

word2id,id2word=get_word2id(text_all)
text_ids=sequence_padding(text_all,word2id,max_seq_len)
aspect_ids=sequence_padding(aspect,word2id,max_aspect_len)
text_left_ids=sequence_padding(text_left_all,word2id,max_left_len)

embedding_matrix=embedding_matrix(word2id,word2vec)
#将数据处理得到的下游任务需要的部分保存
with open('train.pkl','wb') as outp:
    pickle.dump(polarity,outp)
    pickle.dump(word2id,outp)
    pickle.dump(id2word,outp)
    pickle.dump(embedding_matrix,outp)
    pickle.dump(text_ids,outp)
    pickle.dump(aspect_ids,outp)
    pickle.dump(text_left_ids,outp)





with codecs.open('/home/hu/NLP/Recurrent Attention Network/datasets/semeval14/Restaurants_Test_Gold.xml.seg','r','utf-8') as inp:
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
            text_string=text_left+lines[i+1].lower().strip()+text_right
            
            text_string=get_word(text_string)            
            
            text_left=get_word(text_left)
#            seq_leni=len(text_string)
#            if max_seq_len<seq_leni:
#               max_seq_len=seq_leni
            
#            left_leni=len(text_left)
#            if max_left_len<left_leni:
#               max_left_len=left_leni
            #aspect信息
            aspect_i=get_word(lines[i+1].lower().strip())
#            aspect_leni=len(aspect_i)
#            if max_aspect_len<aspect_leni:
#                max_aspect_len=aspect_leni
            #情感极性
            polarity_i.append(int(lines[i+2].strip())+1)
            aspect_t.append(aspect_i)
            polarity_t.extend(polarity_i)
            text_left_all_t.append(text_left)
            text_all_t.append(text_string)
#print(text_all_t)

text_ids_t=sequence_padding(text_all_t,word2id, max_seq_len)
#print(text_ids_t)
aspect_ids_t=sequence_padding(aspect_t,word2id,max_aspect_len)
text_left_ids_t=sequence_padding(text_left_all_t,word2id,max_left_len)

with open('test.pkl','wb') as outp:
    pickle.dump(polarity_t,outp)
    pickle.dump(text_ids_t,outp)
    pickle.dump(aspect_ids_t,outp)
    pickle.dump(text_left_ids_t,outp)




















