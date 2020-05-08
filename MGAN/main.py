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
from mgan import MGAN
import torch.utils.data as D

#GPU
use_cuda=torch.cuda.is_available()
device=torch.device('cuda' if use_cuda else 'cpu')
torch.manual_seed(1)
if use_cuda:
    torch.cuda.manual_seed(1)


with open('.../train.pkl','rb') as inp:
    polarity=pickle.load(inp)
    word2id=pickle.load(inp)
    id2word=pickle.load(inp)
    embedding_matrix=pickle.load(inp)
    text_ids=pickle.load(inp)
    aspect_ids=pickle.load(inp)
    text_left_ids=pickle.load(inp)

with open('.../test.pkl','rb') as inp:
    polarity_t=pickle.load(inp)
    text_ids_t=pickle.load(inp)
    aspect_ids_t=pickle.load(inp)
    text_left_ids_t=pickle.load(inp)    
#将训练数据切分成train和validatiion
#train数据
train_text=text_ids[:int(0.7*len(text_ids))]
train_aspect=aspect_ids[:int(0.7*len(text_ids))]
train_text_left=text_left_ids[:int(0.7*len(text_ids))]
train_polarity=polarity[:int(0.7*len(text_ids))]
#print('train_text',train_text)
#validation数据
validation_text=text_ids[int(0.7*len(text_ids)):]
validation_aspect=aspect_ids[int(0.7*len(text_ids)):]
validation_text_left=text_left_ids[int(0.7*len(text_ids)):]
validation_polarity=polarity[int(0.7*len(text_ids)):]

#test数据
test_text=text_ids
test_aspect=aspect_ids
test_text_left=text_left_ids
test_polarity=polarity

#进行batch
train_text=torch.LongTensor(train_text)
train_aspect=torch.LongTensor(train_aspect)
train_text_left=torch.LongTensor(train_text_left)
train_polarity=torch.LongTensor(train_polarity)

train_dataset=D.TensorDataset(train_text,train_aspect,train_text_left,train_polarity)
train_dataloader=D.DataLoader(train_dataset,Args.Batch,num_workers=2)
#=====
validation_text=torch.LongTensor(validation_text)
validation_aspect=torch.LongTensor(validation_aspect)
validation_text_left=torch.LongTensor(validation_text_left)
validation_polarity=torch.LongTensor(validation_polarity)

validation_dataset=D.TensorDataset(validation_text,validation_aspect,validation_text_left,validation_polarity)
validation_dataloader=D.DataLoader(validation_dataset,Args.Batch,num_workers=2,shuffle=True)
#====================
test_text=torch.LongTensor(test_text)
test_aspect=torch.LongTensor(test_aspect)
test_text_left=torch.LongTensor(test_text_left)
test_polarity=torch.LongTensor(test_polarity)

test_dataset=D.TensorDataset(test_text,test_aspect,test_text_left,test_polarity)
test_dataloader=D.DataLoader(test_dataset,Args.Batch,num_workers=2,shuffle=True)
    

#====================================================
model=MGAN(embedding_matrix).to(device)
criterion=nn.CrossEntropyLoss(size_average=True)

def train(model,train_dataloader):
    model.train()
    optimizer=torch.optim.Adam(model.parameters(),lr=Args.learning_rate,weight_decay=1e-5)
#    acc=0
#    total=0
    result=[]
    labels=[]
    for step,batch_data in enumerate(train_dataloader):
        text_i,aspect_i,text_left_i,polarity_i=batch_data
        text_i=text_i.to(device=device,non_blocking=True)
        aspect_i=aspect_i.to(device=device,non_blocking=True)
        text_left_i=text_left_i.to(device=device,non_blocking=True)
        polarity_i=polarity_i.to(device=device,non_blocking=True)
        y=model(text_i,aspect_i,text_left_i)
        r = np.argmax(y.data.cpu().numpy(),axis=1)
        result.extend(r)
        polarity=polarity_i.cpu().numpy()
        labels.extend(polarity)
        loss=criterion(y,polarity_i)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return result,labels


def validation(model,validation_dataloader):
    model.eval()
    result=[]
    labels=[]
    with torch.no_grad():
        for step,batch_data in enumerate(validation_dataloader):
            text_i,aspect_i,text_left_i,polarity_i=batch_data
            text_i=text_i.to(device=device,non_blocking=True)
            aspect_i=aspect_i.to(device=device,non_blocking=True)
            text_left_i=text_left_i.to(device=device,non_blocking=True)
            polarity_i=polarity_i.to(device=device,non_blocking=True)
            answer=model(text_i,aspect_i,text_left_i)
            
            r=np.argmax(answer.data.cpu().numpy(),axis=1)
            result.extend(r)
            polarity=polarity_i.cpu().numpy()
            labels.extend(polarity)
            
    return result,labels


def test(model,test_dataloader):
    model.eval()
    result=[]
    with torch.no_grad():
       for step,batch_data in enumerate(test_dataloader):
            text_i,aspect_i,text_left_i,polarity_i=batch_data
            text_i=text_i.to(device=device,non_blocking=True)
            aspect_i=aspect_i.to(device=device,non_blocking=True)
            text_left_i=text_left_i.to(device=device,non_blocking=True)
            polarity_i=polarity_i.to(device=device,non_blocking=True)
            answer=model(text_i,aspect_i,text_left_i)
            
            r=np.argmax(answer.data.cpu().numpy(),axis=1)
            result.extend(r)
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
    result1,labels1=train(model,train_dataloader)
    result,labels=validation(model,validation_dataloader)
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
        d_num=0
        torch.save(model.state_dict(),'model')
        print('model is save')

np.set_printoptions(threshold=10000000)
model.load_state_dict(torch.load('model'))
result=test(model,test_dataloader)
r=np.array(result)

file=open('result.txt','w')
file.write(str(r))
file.close()


























