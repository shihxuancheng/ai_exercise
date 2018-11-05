#%%
#%matplotlib inline
import os
import jieba
import numpy as np
import pandas as pd
from wordcloud import WordCloud
from matplotlib import pyplot as plt
import json
from collections import Counter

jieba.set_dictionary('resources/dict.txt.big')

df_qa = pd.read_json('raw_data.json',encoding='utf8')
df_question = df_qa[['question','ans']].copy()
df_question.drop_duplicates(inplace=True)
df_question

all_terms = []
def preProcess(item):
    terms = [t for t in jieba.cut(item,cut_all=True)]
    all_terms.extend(terms)
    return terms

df_question['processed'] = df_question['question'].apply(preProcess)
# df_question
termIndex = list(set(all_terms))
# print(termIndex)

docLen = len(df_question)
# docLen
idfVector = []
for term in termIndex:
    numOfDoc = 0
    for terms in df_question['processed']:
        if term in terms:
            numOfDoc+=1
    idf = np.log(docLen / numOfDoc)
    idfVector.append(idf)

# print(idfVector)

def terms_to_vector(terms):
    vector = np.zeros_like(termIndex,dtype=np.float32)
    for term in terms:
        if term in termIndex:
            idx = termIndex.index(term)
            vector[idx] += 1
    vector = vector * idfVector
    return vector
df_question['vector'] = df_question['processed'].apply(terms_to_vector)

#產生文字雲
def showWordCloud(termList):
    wordcloud = WordCloud(background_color="white",font_path="resources/simsun.ttf",margin=2)
    wordcloud.generate_from_frequencies(frequencies=Counter(termList))
    plt.figure(figsize=(15,15))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()   

showWordCloud(termIndex)

df_question

