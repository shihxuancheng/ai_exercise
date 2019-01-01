import jieba
import pandas as pd

def jieba_init():
    jieba.set_dictionary('resources/dict.txt.big')

#載入raw data並轉換為DataFrame
def load_raw_df(path='row_data.json'):
    df_qa = pd.read_json(path,encoding='utf8')
    df_question = df_qa[['question','ans']].copy()
    df_question.drop_duplicates(inplace=True)
    return df_question

#取得停用字
def get_stop_words(path='resources/stops.txt'):
    with open(path,'r',encoding='utf8') as f:
        stops = f.read().split('\n')
    stop_content = " ".join(x.strip() for x in stops)
    stopList = list(set(stop_content.split()))
    return stopList



jieba_init()

if __name__ == '__main__':
    jieba_init()