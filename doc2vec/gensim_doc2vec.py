#%%
import sys
import os
import logging
import jieba
import gensim
import pandas as pd
import numpy as np
from gensim.models.doc2vec import TaggedDocument

# logging information
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

jieba.set_dictionary('resources/dict.txt.big')

df_qa = pd.read_json('raw_data.json',encoding='utf8')
df_question = df_qa[['question','ans']].copy()
df_question.drop_duplicates(inplace=True)

def preProcess(item):
    #停用字
    with open('resources/stops.txt','r',encoding='utf8') as f:
        stops = f.read().split('\n')
    # stops.append('\n')
    # stops.append('\n\n')
    terms = [t for t in jieba.cut(item,cut_all=False) if t not in stops]
    return terms

def get_dataset():
    x_train = []
    for i,text in enumerate(df_question['question']):
        word_list = ' '.join(jieba.cut(text)).encode('utf-8').split(' ')
        l = len(word_list)
        word_list[l - 1] = word_list[l - 1].strip()
        document = TaggedDocument(word_list, tags=[i])
        x_train.append(document)
    return x_train

def train(x_train,epochs=100):
    # PV-DM w/concatenation - window=5 (both sides) approximates paper's 10-word total window size
    model = gensim.models.Doc2Vec(x_train,dm=1, size=100, window=2,hs=0, min_count=1)
    # PV-DBOW  
    # model = gensim.models.Doc2Vec(docs,dm=0, size=100, hs=0, min_count=2)
    # PV-DM w/average
    # model = gensim.models.Doc2Vec(docs,dm=1, dm_mean=1, size=100, window=2, hs=0, min_count=2)

    model.train(x_train,total_examples=model.corpus_count,epochs=epochs)

    model.save('doc2vec/doc2vec.model')

    return model

def eval():
    model_dm = gensim.models.Doc2Vec.load('doc2vec/doc2vec.model')
    test_text = ['請問要申請WDAMS107可以找誰']
    inferred_vector_dm = model_dm.infer_vector(test_text)
    print(inferred_vector_dm)
    sims = model_dm.docvecs.most_similar([inferred_vector_dm], topn=5)
 
    return sims


if __name__ == '__main__':
    x_train = get_dataset()
    model_dm = train(x_train)
    sims = eval()

    for count, sim in sims:
        sentence = x_train[count]
        words = ''
        for word in sentence[0]:
            words = words + word + ' '
        print(words, sim, len(sentence[0]))




