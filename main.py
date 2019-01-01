#%%
#%matplotlib inline
# import os
import jieba
import numpy as np
import pandas as pd
from numpy.linalg import norm
from wordcloud import WordCloud
from matplotlib import pyplot as plt
import json
from collections import Counter

jieba.set_dictionary('resources/dict.txt.big')

df_qa = pd.read_json('raw_data.json',encoding='utf8')
df_question = df_qa[['question','ans']].copy()
df_question.drop_duplicates(inplace=True)
# df_question

all_terms = []
def preProcess(item):
    #停用字
    with open('resources/stops.txt','r',encoding='utf8') as f:
        stops = f.read().split('\n')
    stops.append('\n')
    stops.append('\n\n')

    terms = [t.lower() for t in jieba.cut(item,cut_all=True) if t not in stops]
    all_terms.extend(terms)
    return terms

# 轉換成TF-IDF vector
def terms_to_vector(terms):
    vector = np.zeros_like(termIndex,dtype=np.dtype(float))
    for term in terms:
        if term in termIndex:
            idx = termIndex.index(term)
            vector[idx] += 1
    vector = vector * idfVector
    return vector


#產生文字雲
def showWordCloud(termList):
    wordcloud = WordCloud(background_color="white",font_path="resources/simsun.ttf",margin=2)
    wordcloud.generate_from_frequencies(frequencies=Counter(termList))
    plt.figure(figsize=(15,15))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()   

# 餘弦相似性計算
def cosine_similarity(v1,v2):
    return np.dot(v1,v2) / (norm(v1) * norm(v2))

def lookup(sentence,numOfReturn=5):
    testing_vector = terms_to_vector(preProcess(sentence))  
    score_dict = {}
    for idx, vec in enumerate(df_question['vector']):
        score = cosine_similarity(testing_vector, vec)
        score_dict[idx] = score
    idxs = np.array(sorted(score_dict.items(), key=lambda x:x[1], reverse=True))[:numOfReturn, 0]
    return df_question.loc[idxs,['question','ans']]



df_question['processed'] = df_question['question'].apply(preProcess)
# df_question

termIndex = list(set(all_terms))
# print(termIndex)

docLen = len(df_question)
# docLen

#計算IDF
idfVector = []
for term in termIndex:
    numOfDoc = 0
    for terms in df_question['processed']:
        # print(terms)
        if term in terms:
            numOfDoc+=1
    idf = np.log(docLen / numOfDoc)
    idfVector.append(idf)




df_question['vector'] = df_question['processed'].apply(terms_to_vector)

# showWordCloud(termIndex)

#%%
s1 = df_question.loc[23]
s2 = df_question.loc[24]
print(s1['question'],'和',s2['question'],'的相識度: ',cosine_similarity(s1['vector'],s2['vector']))


#%%
print(norm(df_question.loc[23]['vector']))
print(norm(df_question.loc[24]['vector']))

#%%
df_question

#%%
query = '請問要申請WDAMS107可以找誰?'
# query = input('您的問題是?')
lookup(query)
