#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 19:22:25 2019

@author: hb
"""
import torch
import torch.nn as nn
import numpy as np
import pickle
from args import Args
from ram import RAM
import torch.utils.data as D
from pytorch_pretrained_bert import BertTokenizer, BertModel
#GPU
use_cuda=torch.cuda.is_available()
device=torch.device('cuda' if use_cuda else 'cpu')
torch.manual_seed(1)
if use_cuda:
    torch.cuda.manual_seed(1)

bert_embeds=BertModel.from_pretrained('.../bert-base-uncased')
bert_embeds.eval()
with open('.../train.pkl','rb') as inp:
    polarity=pickle.load(inp)
    text_ids=pickle.load(inp)
    aspect_ids=pickle.load(inp)
    text_left_ids=pickle.load(inp)
    text_len=pickle.load(inp)
    aspect_len=pickle.load(inp)
    left_len=pickle.load(inp)
    
with open('.../test.pkl','rb') as inp:
    polarity_t=pickle.load(inp)
    text_ids_t=pickle.load(inp)
    aspect_ids_t=pickle.load(inp)
    text_left_ids_t=pickle.load(inp) 
    text_len_t=pickle.load(inp)
    aspect_len_t=pickle.load(inp)
    left_len_t=pickle.load(inp)
#将训练数据切分成train和validatiion
#train数据
train_text=text_ids[:int(0.7*len(text_ids))]
train_aspect=aspect_ids[:int(0.7*len(text_ids))]
train_text_left=text_left_ids[:int(0.7*len(text_ids))]
train_polarity=polarity[:int(0.7*len(text_ids))]
train_text_len=text_len[:int(0.7*len(text_ids))]
train_aspect_len=aspect_len[:int(0.7*len(text_ids))]
train_left_len=left_len[:int(0.7*len(text_ids))]
#print('train_text',train_text)
#validation数据
validation_text=text_ids[int(0.7*len(text_ids)):]
validation_aspect=aspect_ids[int(0.7*len(text_ids)):]
validation_text_left=text_left_ids[int(0.7*len(text_ids)):]
validation_polarity=polarity[int(0.7*len(text_ids)):]
validation_text_len=text_len[int(0.7*len(text_ids)):]
validation_aspect_len=aspect_len[int(0.7*len(text_ids)):]
validation_left_len=left_len[int(0.7*len(text_ids)):]
#test数据
test_text=text_ids_t
test_aspect=aspect_ids_t
test_text_left=text_left_ids_t
test_polarity=polarity_t
test_text_len=text_len_t
test_aspect_len=aspect_len_t
test_left_len=left_len_t
#进行batch
train_text=torch.LongTensor(train_text)
train_aspect=torch.LongTensor(train_aspect)
train_text_left=torch.LongTensor(train_text_left)
train_polarity=torch.LongTensor(train_polarity)
train_text_len=torch.LongTensor(train_text_len)
train_aspect_len=torch.LongTensor(train_aspect_len)
train_left_len=torch.LongTensor(train_left_len)

#=====
validation_text=torch.LongTensor(validation_text)
validation_aspect=torch.LongTensor(validation_aspect)
validation_text_left=torch.LongTensor(validation_text_left)
validation_polarity=torch.LongTensor(validation_polarity)
validation_text_len=torch.LongTensor(validation_text_len)
validation_aspect_len=torch.LongTensor(validation_aspect_len)
validation_left_len=torch.LongTensor(validation_left_len)
#====================
test_text=torch.LongTensor(test_text)
test_aspect=torch.LongTensor(test_aspect)
test_text_left=torch.LongTensor(test_text_left)
test_polarity=torch.LongTensor(test_polarity)
test_text_len=torch.LongTensor(test_text_len)
test_aspect_len=torch.LongTensor(test_aspect_len)  
test_left_len=torch.LongTensor(test_left_len)      
#====================================================
model=RAM().to(device)
criterion=nn.CrossEntropyLoss(size_average=True)

def train(model,train_text,train_aspect,train_text_left,train_polarity,train_text_len,train_aspect_len,train_left_len):
    model.train()
    optimizer=torch.optim.Adam(model.parameters(),lr=Args.learning_rate,weight_decay=1e-5)
#    acc=0
#    total=0
    result=[]
    labels=[]
    for i in range(0,len(train_text),Args.Batch):
        if i+Args.Batch<=len(train_text):
            input_train_text = train_text[i:i+Args.Batch]
            input_train_aspect = train_aspect[i:i+Args.Batch]
            input_train_text_left= train_text_left[i:i+Args.Batch]
            input_train_polarity = train_polarity[i:i+Args.Batch]
            text_len=train_text_len[i:i+Args.Batch]
            aspect_len=train_aspect_len[i:i+Args.Batch]
            left_len=train_left_len[i:i+Args.Batch]
        else:
            input_train_text = train_text[i:len(train_text)]
            input_train_aspect = train_aspect[i:len(train_text)]
            input_train_text_left= train_text_left[i:len(train_text)]
            input_train_polarity = train_polarity[i:len(train_text)]
            text_len=train_text_len[i:len(train_text)]
            aspect_len=train_aspect_len[i:len(train_text)]
            left_len=train_left_len[i:len(train_text)]
        for param in bert_embeds.parameters():
            param.requires_grad = False   
        input_train_text,_=bert_embeds(input_train_text)
        input_train_text=input_train_text[0]
        
        input_train_aspect,_=bert_embeds(input_train_aspect)
        input_train_aspect=input_train_aspect[0]
        input_train_text=input_train_text.cuda()
        input_train_aspect=input_train_aspect.cuda()
        input_train_text_left=input_train_text_left.cuda()
        input_train_polarity=input_train_polarity.cuda()
        text_len=text_len.cuda()
        aspect_len=aspect_len.cuda()
        left_len=left_len.cuda()

        y=model(left_len,text_len,aspect_len,input_train_text,input_train_aspect)
        labels.extend(polarity)
        loss=criterion(y,input_train_polarity)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        r = torch.max(y.data,1)[1].cpu().numpy()
        result.extend(r)
        del input_train_text,input_train_aspect,input_train_text_left,input_train_polarity,y
    return result,labels


def validation(model,validation_text,validation_aspect,validation_text_left,validation_polarity,validation_text_len,validation_aspect_len,validation_left_len):
    model.eval()
    result=[]
    labels=[]
    with torch.no_grad():
        for i in range(0,len(validation_text),Args.Batch):
            if i+Args.Batch<=len(validation_text):
                input_validation_text = validation_text[i:i+Args.Batch]
                input_validation_aspect = validation_aspect[i:i+Args.Batch]
                input_validation_text_left= validation_text_left[i:i+Args.Batch]
                input_validation_polarity = validation_polarity[i:i+Args.Batch]
                text_len=train_text_len[i:i+Args.Batch]
                aspect_len=train_aspect_len[i:i+Args.Batch]
                left_len=train_left_len[i:i+Args.Batch]
            else:
                input_validation_text = validation_text[i:len(validation_text)]
                input_validation_aspect = validation_aspect[i:len(validation_text)]
                input_validation_text_left= validation_text_left[i:len(validation_text)]
                input_validation_polarity = validation_polarity[i:len(validation_text)]
                text_len=validation_text_len[i:len(validation_text)]
                aspect_len=validation_aspect_len[i:len(validation_text)]
                left_len=validation_left_len[i:len(validation_text)]                
              
            for param in bert_embeds.parameters():
                param.requires_grad = False
            input_validation_text,_=bert_embeds(input_validation_text)
            input_validation_text=input_validation_text[0]
            
            
            input_validation_aspect,_=bert_embeds(input_validation_aspect)
            input_validation_aspect=input_validation_aspect[0]
            
            
            input_validation_text=input_validation_text.cuda()
            input_validation_aspect=input_validation_aspect.cuda()
            input_validation_text_left=input_validation_text_left.cuda()
            input_validation_polarity=input_validation_polarity.cuda()
            text_len=text_len.cuda()
            aspect_len=aspect_len.cuda()
            left_len=left_len.cuda()
            
            answer=model(left_len,text_len,aspect_len,input_validation_text,input_validation_aspect)
            
            r=np.argmax(answer.data.cpu().numpy(),axis=1)
            result.extend(r)
            polarity=input_validation_polarity.cpu().numpy()
            labels.extend(polarity)
            del input_validation_text,input_validation_aspect,input_validation_text_left,input_validation_polarity,answer
    return result,labels


def test(model,test_text,test_aspect,test_text_left,test_text_len,test_aspect_len,test_left_len):
    model.eval()
    result=[]
    with torch.no_grad():
       for i in range(0,len(test_text),Args.Batch):
            if i+Args.Batch<=len(test_text):
                input_test_text = test_text[i:i+Args.Batch]
                input_test_aspect = test_aspect[i:i+Args.Batch]
                input_test_text_left= test_text_left[i:i+Args.Batch]
                text_len = test_text_len[i:i+Args.Batch]
                aspect_len = test_aspect_len[i:i+Args.Batch]
                left_len = test_left_len[i:i+Args.Batch]
            else:
                input_test_text = test_text[i:len(test_text)]
                input_test_aspect = test_aspect[i:len(test_text)]
                input_test_text_left= test_text_left[i:len(test_text)]
                text_len = test_text_len[i:len(test_text)]
                aspect_len = test_aspect_len[i:len(test_text)]
                left_len = test_left_len[i:len(test_text)]
            for param in bert_embeds.parameters():
                param.requires_grad = False                
            input_test_text,_=bert_embeds(input_test_text)
            input_test_text=input_test_text[0]
                        
            input_test_aspect,_=bert_embeds(input_test_aspect)
            input_test_aspect=input_test_aspect[0]          
            
            input_test_text=input_test_text.cuda()
            input_test_aspect=input_test_aspect.cuda()
            input_test_text_left=input_test_text_left.cuda()  
            text_len=text_len.cuda()
            aspect_len=aspect_len.cuda()
            left_len=left_len.cuda()
            
            answer=model(left_len,text_len,aspect_len,input_test_text,input_test_aspect)           
            r=np.argmax(answer.data.cpu().numpy(),axis=1)
            result.extend(r)
            del input_test_text,input_test_aspect,input_test_text_left,answer
    return result


lst=[]
def lsts2lst(lsts,lst=lst):
    for l in lsts:
        if isinstance(l,list):
            lst=lsts2lst(l,lst)
        else:
            lst.append(l)
    return lst


mean_l=[]
max_acc=0
d_num=0
from sklearn.metrics import f1_score
for i in range(Args.epoch):
    result1,labels1=train(model,train_text,train_aspect,train_text_left,train_polarity,train_text_len,train_aspect_len,train_left_len)
    result,labels=validation(model,validation_text,validation_aspect,validation_text_left,validation_polarity,validation_text_len,validation_aspect_len,validation_left_len)
    r=np.array(result)
    t=np.array(labels)
    r1=[]
    t1=[]
    r1=lsts2lst(r,r1)
    t1=lsts2lst(t,t1)
    acc=0
    for i in range(len(r1)):
        if(t1[i]==r1[i]):
            acc+=1
    acc=acc/len(r1)
    print("acc:",acc)
    f1_s=f1_score(t1,r1,average='macro')
    mean_l.append([acc,f1_s])
    
    lr=Args.learning_rate*0.8
    
    if acc>max_acc:
        max_acc=acc
#        d_num=0
        torch.save(model.state_dict(),'model')
        print('model is save')

np.set_printoptions(threshold=10000000)
model.load_state_dict(torch.load('model'))
result=test(model,test_text,test_aspect,test_text_left,test_text_len,test_aspect_len,test_left_len)
r=np.array(result)

file=open('result.txt','w')
file.write(str(r))
file.close()


























