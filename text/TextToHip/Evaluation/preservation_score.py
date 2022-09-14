#%% library
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 23:16:39 2020

@author: 황호진
"""

import pandas as pd
import numpy as np
from konlpy.tag import Okt

#%% text to token

def split(original_txt, rhyme_txt):
    okt = Okt()
    if type(original_txt) == list:
        temp_or = [okt.pos(i) for i in original_txt]
        temp_rh = [okt.pos(i) for i in rhyme_txt]
    
    elif type(original_txt) == str:
        temp_or = original_txt.split(' ')
        temp_rh = rhyme_txt.split(' ')
        temp_or = [temp_or]
        temp_rh = [temp_rh]
        
    
    return temp_or, temp_rh

#%% token vector

def words_vector(original_token, rhyme_token):
    words_or = []
    words_rh = []

    # Original Text
    for i in range(len(original_token)):
        or_list = []
        for j in original_token[i]:
            or_list.append(j[0])
        words_or.append(or_list)

    # Rhyme Text
    for i in range(len(rhyme_token)):
        rh_list = []
        for j in rhyme_token[i]:
            rh_list.append(j[0])
        words_rh.append(rh_list)
    
    return words_or, words_rh


#%% Preservation Score
def pre_score(original_vector, rhyme_vector):
    words_score = [] # intersection of token
    if len(original_vector) <= len(rhyme_vector):
        for i in range(len(original_vector)):
            score = 0
            for j in original_vector[i]:
                if j in rhyme_vector[i]:
                    score += 1
            words_score.append(score)
        
    else:
         for i in range(len(rhyme_vector)):
             score = 0
             for j in rhyme_vector[i]:
                 if j in original_vector[i]:
                     score += 1
                     
             words_score.append(score)
        
    # counting of token
    rh_len = [len(rhyme_vector[i]) for i in range(len(rhyme_vector))]
    
    score = [round(words_score[i] / rh_len[i], 2) for i in range(len(words_score))]
    final_score = round(np.mean(score),2)
    
    return score, final_score
#%% Run Preservation Score
def final(text1, text2):
    token_or, token_rh = split(text1, text2)
    vector_or, vector_rh = words_vector(token_or, token_rh)
    score, score_mean = pre_score(vector_or, vector_rh)
    
    return score, score_mean


#%% main
def main():
    
    parser = argparse.ArgumentParser(description='Preservation Score') # parser 객체 생성
    parser.add_argument('--lyric', type = str, help = 'directory')
    
    args = parser.parse_args() # 주어진 인자 파싱
    
    # 파일이 들어갔을 때 점수가 산출되게끔 만들기 (워킹 디렉토리 설정)
    # 파일이 나올지, 문장을 넣어서 나올지는 뭐가 더 맞는지 생각하고 scoring 되게끔 해라.
    sentence1 = input('평가하고 싶은 문장을 입력하세요 : ')
    sentence2 = rhyme_ehancement(sentence1)
    
    print(final(sentence1, sentence2))

#%%
    
if __name__ == '__main__':
    main()
