#%%
#%matplotlib inline
import sys
from os.path import abspath, join, dirname
sys.path.insert(0, join(abspath(dirname(".")), 'src'))

import dataset_utils as dUtil
import numpy as np
import pandas as pd
from numpy.linalg import norm
from collections import Counter
df_qa = None

all_terms = []
idfVector = []

#資料預處理
def preProcess():
    global all_terms, idfVector,df_qa
    all_terms.clear()
    idfVector.clear()

    df_qa = dUtil.load_raw_df()
    df_qa['processed'] = df_qa['question'].apply(sentence_to_words)

    for words in df_qa['processed']:     
        all_terms.extend(words)

    all_terms = list(set(all_terms))

    for term in all_terms:
        docLen = len(df_qa)
        numOfDoc = 0
        for terms in df_qa['processed']:
            if term in terms:
                numOfDoc+=1

        idf = np.log(docLen / numOfDoc)
        idfVector.append(idf)

    df_qa['vector'] = df_qa['processed'].apply(terms_to_vector)

    return df_qa
    
#句詞轉換    
def sentence_to_words(sentence):
    words = dUtil.cut_sentence(sentence,cut_all=True)
    return words

# 轉換成BOW vector
def terms_to_vector(terms):
    #TF轉換成BOW(詞袋)
    vector = np.zeros_like(all_terms,dtype=np.dtype(float))
    for term in terms:
        if term in all_terms:
            idx = all_terms.index(term)
            vector[idx] += 1
    # TF * IDF        
    vector = vector * idfVector

    return vector
 

# 餘弦相似性計算
def cosine_similarity(v1,v2):
    return np.dot(v1,v2) / (norm(v1) * norm(v2))

# 比對相似性並回傳最佳答案
def lookup(sentence,numOfReturn=5):
    test_vec = terms_to_vector(sentence_to_words(sentence))  
    score_dict = {}
    for idx, vec in enumerate(df_qa['vector']):
        score = cosine_similarity(test_vec, vec)
        score_dict[idx] = score
    idxs = np.array(sorted(score_dict.items(), key=lambda x:x[1], reverse=True))[:numOfReturn]
    ans = df_qa.loc[idxs[:,0],['question','ans']]
    ans['similarity'] = idxs[:,1]
    
    if (len(ans) > 0) and (ans.values[0][2] > 0):
        for idx,an in enumerate(ans.values):
            print("相似第{id}名: ".format(id=(idx+1)),an[0])
            print('相似度: ',an[2])
            print('答案: ',an[1])
            print("----------------------------")    
    else:
        print('抱歉! 沒有相似的答案')
    # return ans


#%%
# preProcess()
# ans = lookup(u'請問要申請WDAMS107可以找誰?')
# for idx,ans in enumerate(ans.values):
#     print("第{id}名: ".format(id=(idx+1)),ans[0])
#     print('相似度: ',ans[2])
#     # print('答案: ',ans[1])
#     print("----------------------------")

#%%
def main():
    preProcess()
    while True:
        print('請輸入您的問題:')
        try:
            question = input()
            print('問題是: ',dUtil.cut_sentence(question,cut_all=False))
            lookup(question)
        except Exception as e:
            print(repr(e))

if __name__ == '__main__':
    main()