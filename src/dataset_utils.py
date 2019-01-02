import jieba
import pandas as pd
import sys
import sys
from os.path import abspath, join, dirname
sys.path.insert(0, join(abspath(dirname(".")), ''))

def jieba_init():
    jieba.set_dictionary('resources/dict.txt.big')
    print('jieba dict assigned!')


#載入raw data並轉換為DataFrame
def load_raw_df(path='raw_data.json'):
    df_qa = pd.read_json(path,encoding='utf8')
    df_question = df_qa[['question','ans']].copy()
    df_question.drop_duplicates(inplace=True)
    return df_question

# 取得停用字
def get_stop_words(path='resources/stops.txt'):
    with open(path,'r',encoding='utf8') as f:
        stops = f.read().split('\n')
    stops.append('\n')
    stops.append('\n\n')
    stop_content = " ".join(x.strip() for x in stops)
    stopList = list(set(stop_content.split()))
    return stopList

# 分詞
def cut_sentence(sentence,cut_all=False,stop_word=True):
    if stop_word == True:
        stopList = get_stop_words()
        wordList = [word for word in jieba.cut(sentence,cut_all=cut_all) if word not in stopList]
    else:
        wordList = [word for word in jieba.cut(cut_all=cut_all)]

    return wordList


jieba_init()

if __name__ == '__main__':
    print(sys.path)
    jieba_init()