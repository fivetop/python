# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 15:05:46 2020

@author: 고지형, 배진수
"""
#%% library 
import pandas as pd 
from jamo import j2hcj, h2j 
import numpy as np
from numpy import dot
from numpy.linalg import norm
import pickle 
import argparse
import re

#%% function 

def sigma(n):
    output = 0
    for i in range(1, n+1):
        output+=i
    return output

def score(str1, str2):
    if (len(str1)>1) and (len(str2) >1):
        n_windows = max([len(str1)-1, len(str2)-1])
        score_list, output = [], 0
        for step in range(n_windows):
            if step == 0:
                score_list.append((score_each(str1[-2: ], str2[-2:])))
            else:
                if (str1[-2-step:-step] == '') or (str2[-2-step:-step]==''):
                    break
                score_list.append(score_each(str1[-2-step: -step], str2[-2-step: -step]))
        weights = sorted([i for i in range(1, len(score_list)+1)], reverse=True)
        return (np.array(score_list) * np.array(weights)).sum() / sigma(len(score_list))
    else:
        return score_each(str1, str2)






def score_each(str1, str2):
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


# 1등급 모음 그룹: ['ㅏ', 'ㅘ', 'ㅑ'], ['ㅓ', 'ㅕ', 'ㅝ'], ['ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅙ', 'ㅚ', 'ㅞ'], ['ㅗ', 'ㅛ'], ['ㅜ', 'ㅠ', 'ㅡ'], ['ㅟ', 'ㅢ', 'ㅣ']  
# 2등급 모음 그룹: ['ㅏ', 'ㅐ', 'ㅑ,' 'ㅒ,' 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ'], ['ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
# - 2등급 모음: 음성 모음과 양성 모음으로 구별한다.



def score_element(x1, x2):
    ullim_sori = ['ㄴ', 'ㄹ', 'ㅇ', 'ㅁ']
    moum_similar_A = pd.Series([['ㅏ', 'ㅘ', 'ㅑ'], ['ㅓ', 'ㅕ', 'ㅝ'], ['ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅙ', 'ㅚ', 'ㅞ'], ['ㅗ', 'ㅛ'], ['ㅜ', 'ㅠ', 'ㅡ'], ['ㅟ', 'ㅢ', 'ㅣ']])
    moum_similar_B = pd.Series([['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ'], ['ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']])
    score, moum = 0, []
    for x in [x1, x2]:
        if len(x)==3: # 초성, 중성, 종성으로 구성된 문자일 때 모음 append
#             print(x[1])
            moum.append(x[1])
        else:
            moum.append(x[-1])
            
    if moum[0] == moum[1]: # 모음이 완전 같으면 5점
        score += 5
        if len(x1)==len(x2): # 둘다 초중종성 구성이거나 초중성 구성일 때
            score += 3 # 일단 3점 줌
            if (len(x1)==3 and x1[-1] in ullim_sori and x2[-1] not in ullim_sori) or (len(x1)==3 and x1[-1] not in ullim_sori and x2[-1] in ullim_sori): 
                score -= 1 # 둘다 초중종성일 때, 받침 속성이 다르면 1점 감점
        else: # 서로 문자 구성이 다른 경우
            score += 1
        return score
    
    # 모음이 완전 일치하지 않는 경우
    else: 
        det_A = moum_similar_A[moum_similar_A.apply(lambda x: moum[0] in x)].index.values[0]
        det_B = moum_similar_B[moum_similar_B.apply(lambda x: moum[0] in x)].index.values[0]
#         print(det_B)
        if det_A == moum_similar_A[moum_similar_A.apply(lambda x: moum[1] in x)].index.values[0]:
            score += 3 # 두 모음이 같은 1등급 유사 모음군에 속할 경우 3점 득점
            if len(x1)==len(x2): # 두 문자 모두 초중종성 조합이거나 초중성 조합인 경우
                score += 3 # 일단 3점 줌
                if (len(x1)==3 and x1[-1] in ullim_sori and x2[-1] not in ullim_sori) or (len(x1)==3 and x1[-1] not in ullim_sori and x2[-1] in ullim_sori): 
                    score -= 1 # 둘다 초중종성일 때, 받침 속성이 다르면 1점 감점
            else: # 서로 문자 구성이 다른 경우
                score += 1
                
        elif det_B == moum_similar_B[moum_similar_B.apply(lambda x: moum[1] in x)].index.values[0]:
            score += 3
            if len(x1)==len(x2): # 두 문자 모두 초중종성 조합이거나 초중성 조합인 경우
                score += 3 # 일단 3점 줌
                if (len(x1)==3 and x1[-1] in ullim_sori and x2[-1] not in ullim_sori) or (len(x1)==3 and x1[-1] not in ullim_sori and x2[-1] in ullim_sori): 
                    score -= 1 # 둘다 초중종성일 때, 받침 속성이 다르면 1점 감점
                score = score * 0.7
            else: # 서로 문자 구성이 다른 경우
                score += 1
                score = score * 0.7
    return score


def sori(text):
    text_list = np.array(list(text))
    text_list = text_list[np.where(text_list!=' ')]
    decompose = pd.Series(text_list).apply(lambda x: j2hcj(h2j(x))).tolist()

    # 끝소리 규칙
    end_sound =  ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅇ']
    convert_end_sound = {'ㄲ': 'ㄱㄱ', 'ㄳ': 'ㄱㅅ', 'ㄶ':'ㄴㅎ', 'ㄵ': 'ㄴㅈ', 'ㄺ': 'ㄹㄱ', 'ㄻ': 'ㄹㅁ', 'ㄼ': 'ㄹㅂ', 'ㄽ': 'ㄹㅅ', 'ㄾ': 'ㄹㅌ', 'ㄿ': 'ㄹㅍ', 'ㅀ': 'ㄹㅎ', 'ㅄ': 'ㅂㅅ', 'ㅆ': 'ㅅㅅ'}
    end_simplize = {'ㅅ': 'ㄷ', 'ㅈ': 'ㄷ', 'ㅊ': 'ㄷ', 'ㅋ': 'ㄱ', 'ㅌ': 'ㄷ', 'ㅍ': 'ㅂ', 'ㅎ': 'ㄷ', 'ㅅㅅ':'ㄷ'}
    for idx, word in enumerate(decompose):
        if len(word)==3 and word[-1] in convert_end_sound.keys():
            decompose[idx] = word[:-1] + convert_end_sound[word[-1]]
    for again in range(10):
        for idx in range(len(decompose)-1):
            f_idx = idx
            b_idx = f_idx + 1
            forth, back = decompose[f_idx], decompose[b_idx]
            if (back[0]=='ㅇ' and forth[-2:]=='ㄹㅎ') or (back[0]=='ㅇ' and forth[-2:]=='ㄴㅎ'):
                decompose[f_idx] = forth[:-1]
            if back[0]=='ㅇ' and forth[-1] in end_sound and forth[-1] != 'ㅇ': # jong_sung -> end_sound 수정
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
            if (back[0] == 'ㄱ' and forth[-1] == 'ㅅ') or (back[0]=='ㄱ' and forth[-1] == 'ㄱ') or (back[0]=='ㄱ' and forth[-1]=='ㅍ'):
                decompose[f_idx] = forth[:-1]
                decompose[b_idx] = 'ㄲ' + back[1:]
            if back[0]=='ㄱ' and forth[-1]=='ㅂ':
                decompose[b_idx] = 'ㄲ' + back[1:]
            if (back[0] == 'ㅅ' and forth[-1] == 'ㅂ') or (back[0]=='ㅅ' and forth[-1]=='ㅅ') or (back[0]=='ㅅ' and forth[-1]=='ㄱ') or (back[0]=='ㅅ' and forth[-1]=='ㄹ')            or (back[0] == 'ㅅ' and forth[-1] == 'ㅍ'):
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
            if back[0]=='ㄹ' and forth[-1]=='ㄱ':
                decompose[b_idx] = 'ㄴ' + back[1:]
                decompose[f_idx] = forth[:-1] + 'ㅇ'
            if back[0]=='ㄹ' and forth[-1]=='ㄴ':
                decompose[f_idx] = forth[:-1] + 'ㄹ'
            if back[0]=='ㄹ' and forth[-1]=='ㅇ':
                decompose[b_idx] = 'ㄴ' + back[1:]
            if back[0]=='ㅁ' and forth[-1]=='ㄱ':
                decompose[f_idx] = forth[:-1] + 'ㅇ'
            if back[0]=='ㄴ' and forth[-1]=='ㄹ':
                decompose[b_idx] = 'ㄹ' + back[1:]
            if back[0]=='ㅅ' and forth[-1]=='ㄱ':
                decompose[b_idx] = 'ㅆ' + back[1:]
            if back[0]=='ㄹ' and forth[-1]=='ㄱ':
                decompose[f_idx] = forth[:-1] + 'ㅇ'
    for idx, word in enumerate(decompose):
        if len(word)==3 and word[-1] not in end_sound:
            decompose[idx] = word[:-1] + end_simplize[word[-1]]
        elif word[-2:]=='ㅅㅅ':
            decompose[idx] = word[:-2] + 'ㄷ'
        elif word[-2:]=='ㅂㅅ':
            decompose[idx] = word[:-2] + 'ㅂ'
        elif word[-2:]=='ㄴㅎ':
            decompose[idx] = word[:-2] + 'ㄴ'
        elif word[-2:]=='ㄱㅅ':
            decompose[idx] = word[:-2] + 'ㄱ'
        elif word[-2:]=='ㄹㅁ':
            decompose[idx] = word[:-2] + 'ㅁ'
        elif word[-2:]=='ㄹㅂ':
            decompose[idx] = word[:-2] + 'ㅂ'
        elif word[-2:]=='ㄱㄱ':
            decompose[idx] = word[:-2] + 'ㄱ'
        elif word[-2:]=='ㄴㅈ':
            decompose[idx] = word[:-2] + 'ㄴ'
        elif word[-2:]=='ㄹㄱ':
            decompose[idx] = word[:-2] + 'ㄱ'
        elif word[-2:]=='ㄹㅁ':
            decompose[idx] = word[:-2] + 'ㅁ'
        elif word[-2:]=='ㄹㅂ':
            decompose[idx] = word[:-2] + 'ㅂ'
        elif word[-2:]=='ㄹㅅ':
            decompose[idx] = word[:-2] + 'ㄷ'
        elif word[-2:]=='ㄹㅌ':
            decompose[idx] = word[:-2] + 'ㄷ'
        elif word[-2:]=='ㄹㅍ':
            decompose[idx] = word[:-2] + 'ㅂ'
        elif word[-2:]=='ㄹㅎ':
            decompose[idx] = word[:-2] + 'ㄹ'
            
    return decompose


    
#%% most_similar
    
def cos_sim(A, B):
    return dot(A, B)/(norm(A)*norm(B))  

def most_similar(word , wordvector):
    
    fix = wordvector[word]
    temp = []
    for i in wordvector.keys():
        temp.append(cos_sim(fix,wordvector[i]))
    list_ = np.argsort(np.array(temp))[-10:]
    temp = [list(wordvector.keys())[i]   for i in list_]
    
    if word in temp:
        temp.remove(word)
    
    return [list(wordvector.keys())[i]   for i in list_]

#%% rhyme enhancement 

def rhyme_enhancement(input, wordvector):
    
    return_ = [input].copy()
    token = input.split(' ')
    hangul = re.compile('[^ ㄱ-ㅣ가-힣]+')
    score_list = []
    
    for e in range(len(token)-1):
        score_list.append(score(token[e],token[-1]))
        
    candy , thresh = np.argmin(score_list) , np.min(score_list)
    candy_word = token[candy]
    
    token2 = most_similar(candy_word,wordvector)
    token999 = []
    for i in token2:
        if hangul.sub('',i) != "" :
            token999.append(hangul.sub('',i))
    token2 = token999
    score_list2 = []
    
    for e in range(len(token2)):
        score_list2.append(score(candy_word,token2[e]))
        
    candy2 , score_  = np.argmax(score_list2) , np.max(score_list2)
    
    if score_ >= thresh:
        token[candy] = token2[candy2]
        return (' '.join(token))
    
    else:
        return return_[0]
    
    
#%% 
    
def main():
      
    parser = argparse.ArgumentParser(description="Rhyme enhacnement code")
    
    # Model
    parser.add_argument('--pretrained_dir', type = str ,  default = "C:/Users/wlstn/Desktop/temp" ,  help='the directory of pretrained vector pickle file')
    parser.add_argument('--input', type = str , default ="C:/Users/wlstn/Desktop/temp/테스트/복구.txt")
    parser.add_argument('--output', type = str , default = 'C:/Users/wlstn/Desktop/temp/테스트/라임향상.txt')

    args = parser.parse_args()
    hangul = re.compile('[^ ㄱ-ㅣ가-힣]+')

    print('loading pretrained vectors..')
    with open(f'{args.pretrained_dir}/wordvector.pickle','rb') as handle:
        args.wordvector = pickle.load(handle)
        
    f = open(args.input ,  'r' , encoding = 'utf-8')
    input = f.readlines()
    f.close()
    
    f = open(args.output , 'w' , encoding = 'utf-8')
    u = 0
    for i in input:
        print( u , len(input))
        u +=1
        i = i.replace('\n','')
        temp = rhyme_enhancement(  i , args.wordvector)
        f.write(temp + '\n')
    f.close()
    
#%% 
if __name__== '__main__':
    main()


