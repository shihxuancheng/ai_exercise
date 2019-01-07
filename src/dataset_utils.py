import jieba
import pandas as pd
import sys
from gensim import corpora
from collections import defaultdict
import os
from os.path import abspath, join, dirname
sys.path.insert(0, join(abspath(dirname(".")), ''))

def init():
    jieba.set_dictionary('resources/dict.txt.big')
    print('jieba dict assigned!')

    path = './data'
    if not os.path.isdir(path):
        os.mkdir(path)


#載入raw data並轉換為DataFrame
def load_raw_df(path='raw_data.json'):
    df_qa = pd.read_json(path,encoding='utf8')
    df_question = df_qa[['cate','question','ans']].copy()
    df_question.drop_duplicates(inplace=True)
    return df_question

# 取得停用字
def get_stop_words(path='resources/stops.txt'):
    with open(path,'r',encoding='utf8') as f:
        stops = f.read().split('\n')
    # stops.append('\n')
    # stops.append('\n\n')
    stop_content = " ".join(x.strip() for x in stops)
    stopList = list(set(stop_content.split()))
    return stopList

# 分詞
def cut_sentence(sentence,cut_all=False,stop_word=True):
    if stop_word == True:
        stopList = get_stop_words()
        wordList = [word.lower() for word in jieba.cut(sentence,cut_all=cut_all) if word not in stopList]
    else:
        wordList = [word.lower() for word in jieba.cut(cut_all=cut_all)]

    return wordList


#建立語料庫
def build_corpora(contents,dict_path,corp_save_path,stoplist=[],skip_once=True):
    dict = load_dictionary(dict_path)
    texts = [[word for word in doc if word not in stoplist] for doc in contents]
    if skip_once:
        feq = defaultdict(int)
        for text in texts:
            for token in text:
                feq[token] += 1
        texts = [[token for token in text if feq[token] > 1] for text in texts]
    corpus = [dict.doc2bow(text) for text in texts]            
    corpora.MmCorpus.serialize(corp_save_path, corpus)

#讀取語料庫
def load_corpora(corps_path):
    corpus = corpora.MmCorpus(corps_path)
    return corpus

#建立字典檔
def build_dictionary(contents, dicsavepath, stoplist = []):    
    dict = corpora.Dictionary(contents)
    stop_ids = [dict.token2id[stopword] for stopword in stoplist if stopword in dict.token2id]
    # print('stop id len',len(stop_ids)
    dict.filter_tokens(stop_ids)
    dict.compactify() 
    dict.save(dicsavepath)
    print(dict)

#讀取字典檔
def load_dictionary(dicsavepath):
    dict = corpora.Dictionary.load(dicsavepath)    
    return dict 

init()
if __name__ == '__main__':
    init()