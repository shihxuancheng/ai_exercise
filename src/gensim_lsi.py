#%%
import os
import logging
import sys
from os.path import abspath, join, dirname
sys.path.insert(0, join(abspath(dirname(".")), 'src'))

import dataset_utils as dUtil
from gensim import corpora, models, similarities
from gensim_model import build_lsi_model, load_lsi_model,load_lad_model,load_tfidf_model,load_rp_model

lsi_path = 'data/raw_data.lsi'
tfidf_path = 'data/raw_data.tfidf'
dict_path = 'data/raw_data.dict'
corpora_path = 'data/raw_data.mm'
index_lsi_path ='data/raw_data.lsi.index' 
df_qa = None

def preProcess():
    global questions,df_qa
    df_qa = dUtil.load_raw_df()

    words = [dUtil.cut_sentence(sentence,cut_all=False) for sentence in df_qa['question']]
    stoplist = dUtil.get_stop_words()

    #字典檔
    if not os.path.exists(dict_path):
        dUtil.build_dictionary(words,dict_path,stoplist)
    #語料庫
    if not os.path.exists(corpora_path):
        dUtil.build_corpora(words,dict_path,corpora_path,stoplist)

    dict = dUtil.load_dictionary(dict_path)
    corpora = dUtil.load_corpora(corpora_path)

    #lsi model
    if not os.path.exists(lsi_path):
        build_lsi_model(corpora,dict,tfidf_path,lsi_path)

    return df_qa



def lsi_similarity(sentence,dictionary_path=dict_path,corpora_path=corpora_path,tfidf_model_path=tfidf_path,lsi_model_path=lsi_path,index_path=index_lsi_path):
    
    dict = dUtil.load_dictionary(dict_path)
    corpus = dUtil.load_corpora(corpora_path)
    # tfidfmodel = load_tfidf_model(tfidf_model_path)
    lsimodel = load_lsi_model(lsi_model_path)
    # corpus_tfidf = tfidfmodel[corpus]
    

    if os.path.exists(index_path):
        index_sim = similarities.MatrixSimilarity.load(index_path)
    else:
        # index_sim = similarities.MatrixSimilarity(lsimodel[corpus_tfidf])
        index_sim = similarities.MatrixSimilarity(lsimodel[corpus]) 
        index_sim.save(index_path)
        
    vec_bow = dict.doc2bow(dUtil.cut_sentence(sentence,cut_all=False))
    # vectfidf = tfidfmodel[vec_bow]
    # vec_lsi = lsimodel[vectfidf]
    vec_lsi = lsimodel[vec_bow]
    sims = index_sim[vec_lsi]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])

    # print(sims[:5])    
    if (len(sims) > 0) and (sims[0][1] > 0):
        for id,sim in enumerate(sims[:5]):
            index = sim[0]
            distance = sim[1]
            print('相似第{id}名: '.format(id=id+1),df_qa.loc[(index+1)]['question'])
            print('答案: ',df_qa.loc[(index+1)]['ans'])
            print('相似度:', distance)
            print("----------------------------")
    else:
        print('抱歉! 沒有相似的答案')
#%%
# question = u'請問要申請WDAMS107可以找誰?'
# preProcess()
# lsi_similarity(question)

#%%
def main():
    preProcess()
    while True:
        print('請輸入您的問題:')
        try:
            question = input()
            print('問題是: ',dUtil.cut_sentence(question,cut_all=False))
            lsi_similarity(question)
        except Exception as e:
            print(repr(e))
        
if __name__ == '__main__':
    main()