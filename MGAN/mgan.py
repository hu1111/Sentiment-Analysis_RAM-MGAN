#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 18:53:42 2019

@author: hb
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from args import Args
use_cuda=torch.cuda.is_available()
device=torch.device('cuda' if use_cuda else 'cpu')
torch.manual_seed(1)
if use_cuda:
    torch.cuda.manual_seed(1)
    
    
class MGAN(nn.Module):#memeory是text_all的embedding
    def locationed_context(self,context,context_len,left_len,aspect_len):#重点是要有left_len，aspect_len,memory_len
        batch_size=context.shape[0]
        seq_len=context.shape[1]
        context_len=context_len.cpu().numpy()
        context=context.cuda()
        aspect_len=aspect_len.cpu().numpy()
        
        weight=[[] for i in range(batch_size)]
        for i in range(batch_size):
            #对于aspect左边的词
            for idx in range(left_len[i]):
                weight[i].append(1-(left_len[i]-idx)/(context_len[i]-aspect_len[i]+1))
            for idx in range(left_len[i],left_len[i]+aspect_len[i]):
                weight[i].append(0)
            for idx in range(left_len[i]+aspect_len[i],context_len[i]):
                weight[i].append(1-(idx-left_len[i]-aspect_len[i]+1)/(context_len[i]-aspect_len[i]+1))
            for idx in range(context_len[i],seq_len):
                weight[i].append(0)
        weight=torch.tensor(weight).unsqueeze(2).to(device,non_blocking=True)
        context = context.float()
        weight = weight.float()
        context=torch.cat([context*weight],dim=2)
        return context
        
    
    def c_aspect2context(self,context,aspect,nonzeros_aspect):         
        aspect_sum=torch.sum(aspect,dim=1)
        aspect_avg=torch.div(aspect_sum,nonzeros_aspect.unsqueeze(-1))#[100,400]
        aspect_avg=aspect_avg.unsqueeze(1)#[100,1,400]
        w=self.w1.cuda()
        s=torch.matmul(aspect_avg,w)
        context_new=context.permute(0,2,1)#[100,400,38]
        s_all=torch.matmul(s,context_new)#[100,1,38]
        alpha=F.softmax(s_all,dim=2)
        mca=torch.bmm(alpha,context).squeeze(1)#[batch_size,hidden_dim*2]
        return mca
        
    def c_context2aspect(self,context,aspect,nonzeros_context):         
        context_sum=torch.sum(context,dim=1)
        context_avg=torch.div(context_sum,nonzeros_context.unsqueeze(-1))
        context_avg=context_avg.unsqueeze(1)#[100,1,400]
        w=self.w2.cuda()
        s=torch.matmul(context_avg,w)#[100,1,400]
        aspect_new=aspect.permute(0,2,1)#[100,400,5]
        s_all=torch.matmul(s,aspect_new)#[100,1,5]
        alpha=F.softmax(s_all,dim=2)#[100,1,5]
        mcc=torch.bmm(alpha,aspect).squeeze(1)#[100,400]
        return mcc
        
    def f_aspect2context(self,context,aspect,seq_len,aspect_len_all):
        batch_size=context.shape[0]
        #生成关系矩阵
        for k in range(batch_size):
            mfa=[[] for i in range(batch_size)]
            u=torch.zeros(batch_size,seq_len,aspect_len_all)
            s=[[] for i in range(seq_len)]
            for i in range(seq_len):
                for j in range(aspect_len_all):
                    z=self.fc1(torch.cat((context[k][i],aspect[k][j],torch.mul(context[k][i],aspect[k][j]))))
                    u[i][j]=z
                s[i].append(torch.max(u[i,:]))
            alpha=self.softmax(s)
            mfa[k].append(torch.bmm(alpha,context[k]).squeeze(1))            
        return mfa
            
    def f_context2aspect(self,context,aspect,seq_len,aspect_len_all):
        batch_size=context.shape[0]
        #生成关系矩阵
        for k in range(batch_size):
            mfc=[[] for i in range(batch_size)]
            u=torch.zeros(batch_size,seq_len,aspect_len_all)
            s=[[] for i in range(seq_len)]
            for i in range(seq_len):
                for j in range(aspect_len_all):
                    z=self.fc1(torch.cat((context[k][i],aspect[k][j],torch.mul(context[k][i],aspect[k][j]))))
                    u[i][j]=z
                alpha=self.softmax(u[i])
                s[i].append(torch.bmm(alpha,aspect[k]).squeeze(1))
            mfc[k].append(self.avgpool(s).squeeze(1))
        return mfc        
        
    def __init__(self,embedding_matrix):
        super(MGAN,self).__init__()
        self.args=Args
        self.embed=nn.Embedding.from_pretrained(torch.tensor(embedding_matrix,dtype=torch.float))
        self.bilstm=LSTM(Args.embed_dim,Args.hidden_dim)
        self.attn_linear=nn.Linear(Args.hidden_dim*2,1)
        self.gru_cell=nn.GRUCell(Args.hidden_dim*2+1,Args.embed_dim)
        self.dense=nn.Linear(Args.embed_dim,Args.polarities_dim)
#        self.softmax = F.softmax()
        self.w1=torch.randn(Args.hidden_dim*2,Args.hidden_dim*2)
        self.w2=torch.randn(Args.hidden_dim*2,Args.hidden_dim*2)
        self.w3=torch.randn(1,Args.hidden_dim*6)
        self.fc1=nn.Linear(Args.hidden_dim*6,1)
        self.fc2=nn.Linear(Args.hidden_dim*8,Args.num_class)
        self.avgpool=nn.AvgPool2d(1)
        
    def forward(self,text,aspect,left):
        
        seq_len=text.shape[1]
        aspect_len_all=aspect.shape[1]
        left_len=torch.sum(left!=0,dim=-1)
        context_len=torch.sum(text!=0,dim=-1)
        aspect_len=torch.sum(aspect!=0,dim=-1)
        nonzeros_aspect=aspect_len.float()
        nonzeros_context=context_len.float()
        #进行位置权重乘积
        context=self.embed(text)#[100,74,200]
        context,(_,_)=self.bilstm(context,context_len)#[([100, 38, 400])]       
        context=self.locationed_context(context,context_len,left_len,aspect_len)#([100,38,400])
        aspect=self.embed(aspect)#[100,20,200]
        aspect,(_,_)=self.bilstm(aspect,aspect_len)#[100,5,400]
        #粗粒度下的两个attention
        mca=self.c_aspect2context(context,aspect,nonzeros_aspect)
        mcc=self.c_context2aspect(context,aspect,nonzeros_context)
        #细粒度下的两个attention
        mfa=self.f_context2aspect(context,aspect,seq_len,aspect_len_all)
        mfc=self.f_context2aspect(context,aspect,seq_len,aspect_len_all)      
        m=torch.cat((mca,mcc,mfa,mfc),dim=1)
        p=self.softmax(self.fc2(m))

                
        
class LSTM(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(LSTM,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.rnn=nn.LSTM(input_size=input_size,hidden_size=hidden_size,bidirectional=True,batch_first=True)
                
    def forward(self,x,seq_lengths):
        sorted_seq_lengths,indices=torch.sort(seq_lengths,descending=True)
        _,desorted_indices=torch.sort(indices,descending=False)
        x=x[indices]
        x_embed=pack(x,sorted_seq_lengths.cpu().numpy(), batch_first=True)        
        out,(ht,ct)=self.rnn(x_embed)
        
        ht=torch.transpose(ht,0,1)[desorted_indices]
        ht=torch.transpose(ht,0,1)
        
        out=unpack(out,batch_first=True)
        out=out[0]
        out=out[desorted_indices]
        
        ct=torch.transpose(ct,0,1)[desorted_indices]
        ct=torch.transpose(ct,0,1)
        
        return out,(ht,ct)