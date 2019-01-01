#%%
# see logging events 
# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %     (message)s', level=logging.INFO)
import os
import jieba
import pandas as pd
from gensim import corpora, models, similarities
from collections import defaultdict
jieba.set_dictionary('resources/dict.txt.big')
df_qa = pd.read_json('raw_data.json',encoding='utf8')
df_question = df_qa[['question','ans']].copy()
df_question.drop_duplicates(inplace=True)

def preProcess(item):
    #停用字
    # with open('resources/stops.txt','r',encoding='utf8') as f:
    #     stops = f.read().split('\n')
    # stops.append('\n')
    # stops.append('\n\n')
    terms = [t for t in jieba.cut(item,cut_all=False)]
    return terms

# df_question['question'].values.tolist()
df_question['processed'] = df_question['question'].apply(preProcess)

dict = corpora.Dictionary(x for x in df_question['processed'].values.tolist())
# print(dict)

with open('resources/stops.txt','r',encoding='utf8') as f:
    stops = f.read().split('\n')

stop_content = " ".join(x.strip() for x in stops)
# print(stop_content)

stopList = set(stop_content.split())
# print(stopList)

stop_ids = [dict.token2id[stopword] for stopword in stopList if stopword in dict.token2id]
# print(stop_ids)

dict.filter_tokens(stop_ids)
dict.compactify()


texts = [[word for word in doc if word not in stopList] for doc in df_question['processed']]
feq = defaultdict(int)
for text in texts:
    for token in text:
        feq[token] += 1
texts = [[token for token in text if feq[token] > 1] for text in texts]

dict.save('row_data.dict')
# print(dict)

# for word,index in dict.token2id.items():
#     print(word + 'id: ' + str(index))

corpus = [dict.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('row_data.mm',corpus)

#%%
#載入語料庫
if(os.path.exists('row_data.dict')):
    dict = corpora.Dictionary.load('row_data.dict')
    corpus = corpora.MmCorpus('row_data.mm')
else:
    print("No Files!!!")

tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

corpus_tfidf

lsi = models.LsiModel(corpus_tfidf,id2word=dict,num_topics=50)
corpus_lsi = lsi[corpus_tfidf]
lsi.save('row_data.lsi')
corpora.MmCorpus.serialize('row_data_lsi.mm',corpus_lsi)
print('LSI topics:')
lsi.print_topics(50)


#問題相似性計算
#%%
# dict = corpora.Dictionary.load('row_data.dict')
question = '請問要申請WDAMS107可以找誰?'
vec_bow = dict.doc2bow([x for x in jieba.cut(question,cut_all=False)])
print(vec_bow)
vec_lsi = lsi[vec_bow]

# print(vec_lsi)

index = similarities.MatrixSimilarity(lsi[corpus])
index.save('row_data.index')

sims = index[vec_lsi]
sims = sorted(enumerate(sims), key=lambda item: -item[1])
print(sims[:5])

answers = []
for i,line in enumerate(df_question['question'].values):
    answers.append(line)

# 結果輸出
for ans in sims[:5]:
    print("\n相似問題：",  answers[ans[0]])
    print("相似度：", ans[1])

def main():
    print('Main function')

if __name__ == "__main__":
	main()