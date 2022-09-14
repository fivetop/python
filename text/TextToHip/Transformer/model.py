# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 14:29:05 2020

@author: DMQA
"""

#%% model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import copy



class Embedder(nn.Module):
    
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)     
        

    def forward(self, x):
        return self.embed(x)  # a batch fo 64 samples of 15 indices each  and then , print size [64 , 15 , 512]
    
    
class PositionalEncoder(nn.Module):
    
    def __init__(self, args):
        super().__init__()
        self.d_model = args.d_model
        self.max_seq_len = args.fix_len
        self.gpu = args.gpu
        self.gpu_device = args.gpu_device
    
    

        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(self.max_seq_len, self.d_model)
        for pos in range(self.max_seq_len):
            for i in range(0, self.d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/self.d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/self.d_model)))
                
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe) # This is typically used to register a buffer that should not to be considered a model parameter.
        self.pe = pe 
    
    def forward(self, x ):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        
        if self.gpu:
            x = x + (Variable(self.pe[:,:seq_len], requires_grad=False))
            x = x.to(device=self.gpu_device)
        
        else:
            x = x + (Variable(self.pe[:,:seq_len], requires_grad=False))
            
        return x



class MultiHeadAttention(nn.Module):
        
    def __init__(self, args):
        super().__init__()
        self.d_model = args.d_model
        self.d_k = args.d_model // args.heads
        self.h = args.heads
        
        self.q_linear = nn.Linear(args.d_model, args.d_model) # 512의 input을 받아 512로 출력함 . weight of Q size is  512 by h 
        
        self.v_linear = nn.Linear(args.d_model, args.d_model) # 64 15 512 size . 
        self.k_linear = nn.Linear(args.d_model, args.d_model)
        self.dropout = nn.Dropout(args.dropout)
        self.out = nn.Linear(args.d_model, args.d_model) # self.out means weight ^ O   
        

    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        
        # perform linear operation and split into h heads
        
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k) # size maybe 64 * 15* 8 * 64 ( before 64 * 15 * 512)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k) # size maybe 64 * 15* 8 * 64
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k) # size maybe 64 * 15* 8 * 64
        
        # transpose to get dimensions bs * h * sl * d_model
       
        k = k.transpose(1,2)# size maybe 64 * 8* 15 * 64
        q = q.transpose(1,2)# size maybe 64 * 8* 15 * 64
        v = v.transpose(1,2)# size maybe 64 * 8* 15 * 64

        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        
        output = self.out(concat) # before self.out , concat size is  64, 15, 512 
    
        return output # output size is 64,15,512
    
    
def attention(q, k, v, d_k, mask=None, dropout=None):
    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k) #  q : 64 * 8* 15 * 64 , score size is 64, 8, 15, 15
    if mask is not None:
        
        mask = mask.unsqueeze(1)

        scores = scores.masked_fill(mask == 0, -1e9) # 
    scores = F.softmax(scores, dim=-1)   # scores.shape is 64, 8, 15, 15
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v) # output is size 64, 8, 15, 64
    return output


class FeedForward(nn.Module):
    
    def __init__(self, args):
        super().__init__() 
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(args.d_model, args.d_ff)
        self.dropout = nn.Dropout(args.dropout)
        self.linear_2 = nn.Linear(args.d_ff, args.d_model)
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x    
    
    
    

    
class Norm(nn.Module):
    def __init__(self, args):
        super().__init__()
    
        self.size = args.d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = args.eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True))/ (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm
    
    
class EncoderLayer(nn.Module):
    
    def __init__(self, args):
        super().__init__()
        self.norm_1 = Norm(args)
        self.norm_2 = Norm(args)
        self.attn = MultiHeadAttention(args)
        self.ff = FeedForward(args)
        self.dropout_1 = nn.Dropout(args.dropout)
        self.dropout_2 = nn.Dropout(args.dropout)
        
    def forward(self, x, src_msk):
        
        x = x + self.dropout_1(self.attn(x,x,x,src_msk))
        x = self.norm_1(x)
        x = x + self.dropout_2(self.ff(x))
        x = self.norm_2(x)
        return x    
    
    


class DecoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.norm_1 = Norm(args)
        self.norm_2 = Norm(args)
        self.norm_3 = Norm(args)
        
        self.dropout_1 = nn.Dropout(args.dropout)
        self.dropout_2 = nn.Dropout(args.dropout)
        self.dropout_3 = nn.Dropout(args.dropout)
        
        self.attn_1 = MultiHeadAttention(args)
        self.attn_2 = MultiHeadAttention(args)
        self.ff = FeedForward(args)
        
        
    def forward(self, x, e_outputs, src_msk, trg_msk):
            x = x + self.dropout_1(self.attn_1(x, x, x, trg_msk))
            x = self.norm_1(x)
            x = x + self.dropout_2(self.attn_2(x, e_outputs, e_outputs,src_msk)) # mask self attention
            x = self.norm_2(x)
            x = x + self.dropout_3(self.ff(x))
            
            x = self.norm_3(x)
            return x

        
        
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.N = args.N
        self.embed = Embedder(args.src_vocab , args.d_model)
        self.pos = PositionalEncoder(args)
        self.layers = get_clones(EncoderLayer(args), args.N)
        self.norm = Norm(args)
    def forward(self, src, src_msk):
        x = self.embed(src)
        x = self.pos(x)
        for i in range(self.N):
            x = self.layers[i](x, src_msk)
        return x
    
class Decoder(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.N = args.N
        self.embed = Embedder(args.trg_vocab , args.d_model)
        self.pos = PositionalEncoder(args)
        self.layers = get_clones(DecoderLayer(args), args.N)
        self.norm = Norm(args)
    def forward(self, trg, e_outputs, src_msk, trg_msk):
        x = self.embed(trg)
        x = self.pos(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_msk, trg_msk)
        return x
    
    
class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        self.out = nn.Linear(args.d_model, args.trg_vocab)
    def forward(self, src, trg, src_msk, trg_msk):
        e_outputs = self.encoder(src, src_msk)
        d_output = self.decoder(trg, e_outputs, src_msk, trg_msk)
        output = self.out(d_output)
        return output

