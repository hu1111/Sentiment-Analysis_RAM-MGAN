#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 19:23:18 2019

@author: hb
"""
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from args import Args

use_cuda=torch.cuda.is_available()
device=torch.device('cuda' if use_cuda else 'cpu')
torch.manual_seed(1)
if use_cuda:
    torch.cuda.manual_seed(1)

class RAM(nn.Module):#memeory是text_all的embedding
    def locationed_memory(self,memory,memory_len,left_len,aspect_len):#重点是要有left_len，aspect_len,memory_len
        batch_size=memory.shape[0]
        seq_len=memory.shape[1]
        memory_len=memory_len.cpu().numpy()
        memory=memory.cuda()
        aspect_len=aspect_len.cpu().numpy()        
        weight=[[] for i in range(batch_size)]
        u=[[] for i in range(batch_size)]
        for i in range(batch_size):
            #对于aspect左边的词
            for idx in range(left_len[i]-1):
                weight[i].append(1-(left_len[i]-idx-1)/memory_len[i])
                u[i].append(idx-left_len[i]+1)
            for idx in range(left_len[i]-1,left_len[i]+aspect_len[i]-3):
                weight[i].append(1)
                u[i].append(0)
            for idx in range(left_len[i]+aspect_len[i]-3,memory_len[i]):
                weight[i].append(1-(idx-left_len[i]-aspect_len[i]+4)/memory_len[i])
                u[i].append(idx-left_len[i]-aspect_len[i]+4)
            for idx in range(memory_len[i],seq_len):
                weight[i].append(0)
                u[i].append(1)
        u=torch.tensor(u).unsqueeze(2).to(device,non_blocking=True)
        weight=torch.tensor(weight).unsqueeze(2).to(device,non_blocking=True)
        memory = memory.float()
        weight = weight.float()
        u = u.float()
        memory=torch.cat([memory*weight,u],dim=2)
        return memory
    
    def __init__(self):
        super(RAM,self).__init__()
        self.args=Args
        self.bilstm_context=LSTM(Args.embed_dim,Args.hidden_dim)
        self.attn_linear=nn.Linear(Args.hidden_dim*2+1+Args.embed_dim*2,1)
        self.gru_cell=nn.GRUCell(Args.hidden_dim*2+1,Args.embed_dim)
        self.dense=nn.Linear(Args.embed_dim,Args.polarities_dim)
        self.softmax = nn.Softmax()
        
    def forward(self,left_len,memory_len,aspect_len,text,aspect):
        nonzeros_aspect=aspect_len.float()
        
        memory,(_,_)=self.bilstm_context(text,memory_len)
        memory=self.locationed_memory(memory,memory_len,left_len,aspect_len)
        
        aspect=torch.sum(aspect,dim=1)
        nonzeros_aspect = nonzeros_aspect.to(device,non_blocking=True)
        aspect=torch.div(aspect,nonzeros_aspect.unsqueeze(-1)).to(device,non_blocking=True)#在最后一个位置增加一个维度
        et=torch.zeros_like(aspect).to(device)#[8,100]
        
        batch_size=memory.size(0)
        seq_len=memory.size(1)
        for _ in range(self.args.epoch):
            g=self.attn_linear(torch.cat([memory,
                                          torch.zeros(batch_size,seq_len,self.args.embed_dim).to(device)+et.unsqueeze(1),
                                          torch.zeros(batch_size,seq_len,self.args.embed_dim).to(device)+aspect.unsqueeze(1)],
                                           dim=-1))
            alpha=self.softmax(g)
            i=torch.bmm(alpha.transpose(1,2),memory).squeeze(1)
            et=self.gru_cell(i,et)
        out=self.dense(et)
        return out
            

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
        






















        
    
    








        