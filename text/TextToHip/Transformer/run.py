# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 13:41:53 2020

@author: Jinsoo
"""

#%% import 

from Loader import DataLoader
from model import Transformer
import argparse
import time
import gc
import pickle
import numpy as np 
import os 

from optimizer import CosineWithRestarts
import torch.optim 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#%% train 

def train_model(args, train_loss_list , val_loss_list):
    
    best_loss = np.inf

    start = time.time()
    args.model.train()
    temp = start
    
    
    
    for epoch in range(args.epochs):
        
        total_loss = 0

        
        for i, batch in enumerate(args.Load.train_iter):
            
            args.model.train()
            gc.collect()
            src = batch.src[0] # src.shape is [64,15]
            trg = batch.tgt[0] # srg.shape is [64,15]

            trg_input = trg[:, :-1]
            targets = trg[:, 1:].contiguous().view(-1)
            input_pad = args.Load.src.vocab.stoi['<pad>']
            src_msk = ((src != input_pad).unsqueeze(1))
            
            
            # create mask as before
            target_pad = args.Load.tgt.vocab.stoi['<pad>']
            target_msk = (trg_input != target_pad).unsqueeze(1)
            size = trg_input.size(1) # get seq_len for matrix
            nopeak_mask = np.triu(np.ones((1, size, size)),k=1).astype('uint8')
            
            if args.gpu:
                nopeak_mask = Variable(torch.from_numpy(nopeak_mask) == 0).to(args.gpu_device)

            else:
                nopeak_mask = Variable(torch.from_numpy(nopeak_mask) == 0)
            trg_msk = target_msk & nopeak_mask

            
            preds = args.model(src, trg_input, src_msk, trg_msk)


            args.optimizer.zero_grad()            
                  
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), targets , ignore_index=target_pad)
            loss.backward()
            args.optimizer.step()
            
            if args.SGD == True:
                args.sched.step()
                
            
            
            total_loss += loss.data.item()
            if (i+1) % args.print_every ==0:
                                                                
                with torch.no_grad():
                    args.model.eval()
                    val_loss = 0
                    for j, valid_batch in enumerate(args.Load.valid_iter):
                
                        src = valid_batch.src[0] # src.shape is [64,15]
                        trg = valid_batch.tgt[0] # srg.shape is [64,15]

                        trg_input = trg[:, :-1]
                        targets = trg[:, 1:].contiguous().view(-1)
                        input_pad = args.Load.src.vocab.stoi['<pad>']
                        src_msk = ((src != input_pad).unsqueeze(1))


                        # create mask as before
                        target_pad = args.Load.tgt.vocab.stoi['<pad>']
                        target_msk = (trg_input != target_pad).unsqueeze(1)
                        size = trg_input.size(1) # get seq_len for matrix
                        nopeak_mask = np.triu(np.ones((1, size, size)),k=1).astype('uint8')


                        if args.gpu:
                            nopeak_mask = Variable(torch.from_numpy(nopeak_mask) == 0).to(args.gpu_device)

                        else:
                            nopeak_mask = Variable(torch.from_numpy(nopeak_mask) == 0)

                        trg_msk = target_msk & nopeak_mask


                        preds = args.model(src, trg_input, src_msk, trg_msk)
                        v_loss = F.cross_entropy(preds.view(-1, preds.size(-1)), targets , ignore_index=target_pad)
                        val_loss +=v_loss.data.item()
                        
                        
                loss_avg = total_loss / args.print_every
                
                if best_loss > (val_loss/ len(args.Load.valid_iter)):
                    best_loss = (val_loss/ len(args.Load.valid_iter))
                    torch.save(args.model.state_dict(), f"{args.save_dir}/{args.experiment}_best_transformer.pth")
                    
                print("time = %dm, epoch %d, iter = %d, loss = %.3f, %ds per %d iters" % ((time.time() - start) // 60,epoch + 1, i + 1, loss_avg, time.time() - temp,args.print_every))
                print('val loss: {:.4f}'.format(val_loss/ len(args.Load.valid_iter)))
                train_loss_list.append(loss_avg)
                val_loss_list.append(val_loss/ len(args.Load.valid_iter))
                total_loss = 0 
                temp = time.time()
                
                
                
                with open(f'{args.save_dir}/{args.experiment}_train_loss.pickle','wb') as f:
                    pickle.dump(train_loss_list,f)
                with open(f'{args.save_dir}/{args.experiment}_val_loss.pickle','wb') as f:
                    pickle.dump(val_loss_list,f)
                
                
                
        print("%dm: epoch %d [%s%s]  %d%%  loss = %.3f\nepoch %d complete, loss = %.03f" %\
              ((time.time() - start)//60, epoch + 1, "".join('#'*(100//5)), "".join(' '*(20-(100//5))), 100, loss_avg, epoch + 1, loss_avg))                       
                
                
        torch.save(args.model.state_dict(),f'{args.save_dir}/{args.experiment}_epoch{epoch+1}.pth')


def get_len(train):
    
    for i, b in enumerate(train):
        pass
    
    return i

#%% main 








def main():
    
    # args.Loader
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default="D:/프로젝트/비타민/Data_preprocess/비타민팀프로젝트훈련/", help="train_data_directory/")
    parser.add_argument("--validation_dir", default="D:/프로젝트/비타민/Data_preprocess/비타민팀프로젝트검증/", help="validation_data_directory/")
    parser.add_argument('--X' , default = '코어.txt', type = str , help = 'transforemr X')
    parser.add_argument('--Y' , default = '원본.txt', type = str , help = 'transforemr Y')
    parser.add_argument('--fix_len' , default = 15 , type = int , help = 'the fixed length of Token in a sentence')  
    parser.add_argument('--max_len' , default = 25 , type = int , help = 'the fixed length of Token in a sentence')  
    parser.add_argument('--max_vocab' , default = 10000, type = int , help = 'the Maximum token in the vocabulary')
    
    # Model 
    parser.add_argument('--d_ff' , default = 2048 , type = int, help = 'Transformer : d_ff')
    parser.add_argument('--d_model' , default = 512 , type = int , help = 'Embedding vector size')
    parser.add_argument('--heads' , default = 8, type = int, help = 'the number of Multi-Head attention Count')
    parser.add_argument('--dropout' , default = 0.1 , type = float , help = 'Dropout p')
    parser.add_argument("--N" , default = 6 , type = int , help = 'The number of Layder in Encoding or Decoding Block')
    parser.add_argument('--eps' , default = 1e-6 , type = float , help = 'Layer normalization eps')

    # GPU
    parser.add_argument('--gpu' , default = True, action ='store_true' , help ='Where to apply GPU')
    parser.add_argument("--gpu_device", default=0 , type=int, help="the number of gpu to be used")

    # Train
    parser.add_argument('--epochs' , default = 1 , type = int  , help = 'Train epoch number')
    parser.add_argument('--print_every' , default  = 100 , type = int , help = 'print every per % iteration')
    parser.add_argument('--lr' , default = 0.0001 , type = float , help = 'learning rate')
    parser.add_argument('-warmup','--n_warmup_steps', type=int, default=4000)
    parser.add_argument('--batch_size', type=int, default= 128 )
    parser.add_argument('--SGD' , default  = True , action = 'store_true')

    
    
    # Save
    parser.add_argument('--save_dir' , type = str , help = 'save_directory')
    parser.add_argument('--experiment' , type = str , help = 'Experiment number')
    

    args = parser.parse_args()
    args.Load = DataLoader(train_fn = args.train_dir ,
                           max_length = args.max_len,
                           valid_fn=args.validation_dir, 
                           device = args.gpu_device, 
                           exts = (args.X ,args.Y) ,
                           fix_length=args.fix_len,max_vocab = args.max_vocab , batch_size = args.batch_size)
    args.src_vocab = len(args.Load.src.vocab)
    args.trg_vocab = len(args.Load.tgt.vocab)
    
    print("=" * 20 + "Data Loading finied" + "=" * 20)
    print('src_vocab size:', args.src_vocab)
    print('trg_vocab size:', args.trg_vocab)
    
    model = Transformer(args)
    
    
    
    if args.gpu:
        model = model.to(args.gpu_device)
        print('model created !! - gpu version')
        
    else:
        print('model created !! - cpu version')
        
    for p in model.parameters():
        if p.dim() > 1 :
            nn.init.xavier_uniform_(p)
            
    args.model = model
    args.train_len = get_len(args.Load.train_iter)
    args.optimizer = torch.optim.Adam(args.model.parameters() , lr = args.lr,    betas = (0.9, 0.98) , eps = 1e-09 )
    
    if args.SGD == True:
        args.sched = CosineWithRestarts(args.optimizer , T_max = args.train_len)
    

    
    
    
    


    train_loss_list = [] 
    val_loss_list = []
    
# =============================================================================
#     os.mkdir(f'{args.save_dir}')
#     
#     print(f'{args.save_dir} folder created !! ')
#     
# =============================================================================
    
    print('Learning Start')
    train_model(args , train_loss_list , val_loss_list)
    
    



#%% run 
if __name__ == "__main__":
    main()