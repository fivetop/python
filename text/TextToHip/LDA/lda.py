# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 23:34:56 2020

@author: Jinsoo
"""



#%% library 
import pickle 
import re
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import argparse

#%% functions

def load(args):
    
    with open(f'{args.data_dir}data.pickle','rb') as f :
        data = pickle.load(f)
    
    data = data.song_lyric_wd
    
    return data 

def preprocess(data):
    
    temp = [] 
    
    for i in data:
        temp.append(str(i).split('\n'))
    
    temp2 = [] 
    
    for i in temp:
        temp2.append(" ".join(i))
        
    for ind, value in enumerate(temp2):
        p = re.compile("[^0-9]")
        temp2[ind] = "".join(p.findall(value))

    temp3 = []
    
    for i in temp2:
        temp3.append(i.split(' '))
        
    temp3 = pd.Series(temp3)
    
    count_vectorizer = CountVectorizer()

    temp4 = [] 
    
    for i in temp3:
        temp4.append(' '.join(i))
        
    return count_vectorizer.fit_transform(temp4) , count_vectorizer
    
def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))


#%% main

def main():
    
    # args.Loader
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="D:/프로젝트/비타민/", help="train_data_directory/")
    parser.add_argument("--number_topics" , default = 15 , type = int , help ='LDA hyperparameter, topick number')
    parser.add_argument("--n_jobs" , default = -1 , type = int, help ='the number of used core in LDA : default is maximum')
    parser.add_argument("--number_words" , default = 20 , type = int, help ='the number of words in LDA topics')

    args = parser.parse_args()

    data = load(args)
    count_data , count_vectorizer = preprocess(data)
    print('--start LDA processing! --....')
    lda = LDA(n_components= args.number_topics , n_jobs = args.n_jobs )
    lda.fit(count_data)
    print('--finished LDA processing!')
    
    print_topics(lda, count_vectorizer , args.number_words)
    

#%% 
if __name__ == '__main__':
    main()