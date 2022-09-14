#%% library 
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 22:10:57 2020

@author: 고지형, 황호진
"""

from numpy import dot
from numpy.linalg import norm
import numpy as np
import pickle 
import argparse
import pandas as pd
from jamo import j2hcj, h2j
from konlpy.tag import Okt 
import argparse
import itertools

#%% score
def score(str1, str2):
    str1, str2 = sori(str1), sori(str2)
    # 한글자 짜리
    first_score, last_score = 0, 0
    if len(str1)==1 or len(str2)==1:
        temp1, temp2 = str1[-1], str2[-1]
        last_score = score_element(temp1, temp2)
        return last_score
    else:
        temp1, temp2, moum = str1[-1], str2[-1], [] # 마지막 글자
        last_score = score_element(temp1, temp2)
        if last_score == 0:
            return last_score
        else:
            temp1, temp2, moum = str1[-2], str2[-2], []
            first_score = score_element(temp1, temp2)
    return first_score*0.25 + last_score*0.75

#%% score_element
def score_element(x1, x2):
    ullim_sori = ['ㄴ', 'ㄹ', 'ㅇ', 'ㅁ']
    moum_similar = pd.Series([['ㅏ', 'ㅘ', 'ㅑ'], ['ㅓ', 'ㅕ', 'ㅝ'], ['ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅙ', 'ㅚ', 'ㅞ'], ['ㅗ', 'ㅛ'], ['ㅜ', 'ㅠ', 'ㅡ'], ['ㅟ', 'ㅢ', 'ㅣ']])
    score, moum = 0, []
    for x in [x1, x2]:
        if len(x)==3:
            moum.append(x[1])
        else:
            moum.append(x[-1])
    if moum[0] == moum[1]: # 모음이 완전 같으면 5점
        score += 5
        if len(x1)==len(x2):
            score += 3
            if (len(x1)==3 and x1[-1] in ullim_sori and x2[-1] not in ullim_sori) or (len(x1)==3 and x1[-1] not in ullim_sori and x2[-1] in ullim_sori):
                score -= 1
        else:
            score += 1
        return score
    else: 
        det = moum_similar[moum_similar.apply(lambda x: moum[0] in x)].index.values[0]
        if det == moum_similar[moum_similar.apply(lambda x: moum[1] in x)].index.values[0]: # 모음이 완전 같지는 않지만, 유사한 소리의 모음이라면 2점
            score += 3
            if len(x1)==len(x2):
                score += 3
                if (len(x1)==3 and x1[-1] in ullim_sori and x2[-1] not in ullim_sori) or (len(x1)==3 and x1[-1] not in ullim_sori and x2[-1] in ullim_sori):
                    score -= 1
            else:
                score += 1
        else:
            return score
    return score

#%% sori
def sori(text):
    text_list = np.array(list(text))
    text_list = text_list[np.where(text_list!=' ')]
    decompose = pd.Series(text_list).apply(lambda x: j2hcj(h2j(x))).tolist()

    # 끝소리 규칙
    end_sound =  ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅇ']
    convert_end_sound = {'ㄲ': 'ㄱㄱ', 'ㄳ': 'ㄱㅅ', 'ㄶ':'ㄴㅎ', 'ㄵ': 'ㄴㅈ', 'ㄺ': 'ㄹㄱ', 'ㄻ': 'ㄹㅁ', 'ㄼ': 'ㄹㅂ', 'ㄽ': 'ㄹㅅ', 'ㄾ': 'ㄹㅌ', 'ㄿ': 'ㄹㅍ', 'ㅀ': 'ㄹㅎ', 'ㅄ': 'ㅂㅅ', 'ㅆ': 'ㅅㅅ'}
    end_simplize = {'ㅅ': 'ㄷ', 'ㅈ': 'ㄷ', 'ㅊ': 'ㄷ', 'ㅋ': 'ㄱ', 'ㅌ': 'ㄷ', 'ㅍ': 'ㅂ', 'ㅎ': 'ㄷ', 'ㅅㅅ':'ㄷ'}
    for idx, word in enumerate(decompose):
        try:
            if len(word)==3 and word[-1] in convert_end_sound.keys():
                decompose[idx] = word[:-1] + convert_end_sound[word[-1]]
        except:
            print(idx, word)

    for again in range(10):
        for idx in range(len(decompose)-1):
            f_idx = idx
            b_idx = f_idx + 1
            forth, back = decompose[f_idx], decompose[b_idx]
            if (back[0]=='ㅇ' and forth[-2:]=='ㄹㅎ') or (back[0]=='ㅇ' and forth[-2:]=='ㄴㅎ'):
                decompose[f_idx] = forth[:-1]
            elif back[0]=='ㅇ' and forth[-1] in end_sound and forth[-1] != 'ㅇ': # jong_sung -> end_sound 수정
                decompose[f_idx] = forth[:-1]
                decompose[b_idx] = forth[-1] + back[1:]
            if back[0]=='ㅎ' and forth[-1] == 'ㄱ':
                decompose[f_idx] = forth[:-1]
                decompose[b_idx] = 'ㅋ' + back[1:]
            if (back[0]=='ㅈ' and forth[-1] == 'ㄱ') or (back[0]=='ㅈ' and forth[-1]=='ㅂ') or (back[0]=='ㅈ' and forth[-1]=='ㅍ') or (back[0]=='ㅈ' and forth[-1]=='ㄷ'):
                decompose[b_idx] = 'ㅉ' + back[1:]
            if back[0]=='ㅈ' and forth[-2:]=='ㄹㅌ':
                decompose[b_idx] = 'ㅉ' + back[1:]
                decompose[f_idx] = forth[:-1]
            if back[0]=='ㅈ' and forth[-2:]=='ㅅㅅ':
                decompose[f_idx] = forth[:-1]
                decompose[b_idx] = 'ㅉ' + back[1:]
            if (back[0]=='ㄷ' and forth[-1]=='ㅅ') or (back[0]=='ㄷ' and forth[-1]=='ㄷ'):
                decompose[f_idx] = forth[:-1]
                decompose[b_idx] = 'ㄸ' + back[1:]
            if back[0]=='ㄲ' and forth[-2:]=='ㅅㅅ':
                decompose[f_idx] = forth[:-1]
            if back[0]=='ㄱ' and forth[-1]=='ㅎ':
                decompose[f_idx] = forth[:-1]
                decompose[b_idx] = 'ㅋ' + back[1:]
            if (back[0] == 'ㄱ' and forth[-1] == 'ㅅ') or (back[0]=='ㄱ' and forth[-1] == 'ㄱ') or (back[0]=='ㄱ' and forth[-1]=='ㅂ') or (back[0]=='ㄱ' and forth[-1]=='ㅍ'):
                decompose[f_idx] = forth[:-1]
                decompose[b_idx] = 'ㄲ' + back[1:]
            if (back[0] == 'ㅅ' and forth[-1] == 'ㅂ') or (back[0]=='ㅅ' and forth[-1]=='ㅅ') or (back[0]=='ㅅ' and forth[-1]=='ㄱ') or (back[0]=='ㅅ' and forth[-1]=='ㄹ')\
            or (back[0] == 'ㅅ' and forth[-1] == 'ㅍ'):
                decompose[b_idx] = 'ㅆ' + back[1:]
            if back[0]=='ㄷ' and forth[-1]=='ㅎ':
                decompose[f_idx] = forth[:-1]
                decompose[b_idx] = 'ㅌ' + back[1:]
            if back[0]=='ㅎ' and forth[-1]=='ㅅ':
                decompose[b_idx] = 'ㅌ' + back[1:]
                decompose[f_idx] = forth[:-1]
            if back[0]=='ㅎ' and forth[-1]=='ㄷ':
                decompose[b_idx] = 'ㅊ' + back[1:]
            if (back[0]=='ㄱ' and forth[-1]=='ㅅ') or (back[0]=='ㄱ' and forth[-1]=='ㄷ'):
                decompose[b_idx] = 'ㄲ' + back[1:]
            if back[0]=='ㅎ' and forth[-1]=='ㅂ':
                decompose[f_idx] = forth[:-1]
                decompose[b_idx] = 'ㅍ' + back[1:]
            if back[0]=='ㅅ' and forth[-2:]=='ㄴㅈ':
                decompose[b_idx] = 'ㅆ' + back[1:]
                decompose[f_idx] = forth[:-1]
            if (back[0]=='ㅈ' and forth[-2:]=='ㄴㅎ') or (back[0]=='ㅈ' and forth[-1]=='ㅎ'):
                decompose[b_idx] = 'ㅊ' + back[1:]
                decompose[f_idx] = forth[:-1]
            if back[0]=='ㄷ' and forth[-1]=='ㄱ':
                decompose[b_idx] = 'ㄸ' + back[1:]
            if back[0]=='ㅂ' and forth[-1]=='ㄱ':
                decompose[b_idx] = 'ㅃ' + back[1:]
            if back[0]=='ㄷ' and forth[-2:]=='ㄹㅁ':
                decompose[b_idx] = 'ㄸ' + back[1:]
                decompose[f_idx] = forth[:-2] + 'ㅁ'
            if back[0]=='ㄷ' and forth[-2:]=='ㄹㅌ':
                decompose[b_idx] = 'ㄸ' + back[1:]
                decompose[f_idx] = forth[:-2] + 'ㄹ'
    for idx, word in enumerate(decompose):
        if len(word)==3 and word[-1] not in end_sound:
            decompose[idx] = word[:-1] + end_simplize[word[-1]]
        if word[-2:]=='ㅅㅅ':
            decompose[idx] = word[:-2] + 'ㄷ'
        if word[-2:]=='ㅂㅅ':
            decompose[idx] = word[:-2] + 'ㅂ'
        if word[-2:]=='ㄴㅎ':
            decompose[idx] = word[:-2] + 'ㄴ'
        if word[-2:]=='ㄱㅅ':
            decompose[idx] = word[:-2] + 'ㄱ'
        if word[-2:]=='ㄹㅁ':
            decompose[idx] = word[:-2] + 'ㅁ'
    return decompose


#%% All words rhyme score

def rhyme_score(text):
    if type(text) == str:
        temp = text.split(' ')
        list_temp = list(map(' '.join, itertools.permutations(temp, 2)))
    
        result = list()
    
        for i in list_temp:
            split_word = i.split(' ')
            result.append(score(split_word[0], split_word[-1]))
        
        score_result = round(np.mean(result), 2)
        
    else:
        split_list = [text[i].split(' ') for i in range(len(text))]
        
        list_temp = []
        
        for i in range(len(split_list)):
            list_temp.append(list(map(' '.join, itertools.permutations(split_list[i], 2))))
        
        score_list = []
        
        for i in range(len(list_temp)):
            score_list2 = []
            for j in range(len(list_temp[i])):
                score_list2.append(score(list_temp[i][j].split(' ')[0], list_temp[i][j].split(' ')[-1]))
            score_list.append(score_list2)
    
        score_vec = [round(np.nanmean(score_list[i]), 2) for i in range(len(score_list))    ]
        
    
        score_result = round(np.mean(score_vec), 2)
        
        
    return score_result


#%%

def main():
    
    parser = argparse.ArgumentParser(description='Rhyme Score') 
    parser.add_argument('--lyric', type = str, help = 'directory')
    
    args = parser.parse_args() 
    
    sentence = input('평가하고 싶은 문장을 입력하세요 : ')
    
    result = rhyme_score(sentence)
    
    print(result)

#%%
    
if __name__ == '__main__':
    main()
    