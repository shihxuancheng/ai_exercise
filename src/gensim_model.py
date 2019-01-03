import logging

from dataset_utils import load_corpora
from gensim import corpora,similarities,models

#產生TFIDF Model
def tfidf_model(corpus,tfidf_save_path):
    # tfidf = models.TfidfModel(corpus,normalize=True)
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    tfidf.save(tfidf_save_path)
    return corpus_tfidf

#讀取TFIDF Model
def load_tfidf_model(tfidf_save_path):
    model = models.TfidfModel.load(tfidf_save_path)
    return model

#產生LSI Model
def lsi_model(corpus_tfidf,dictionary,lsi_save_path):
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=50)
    lsi.print_topics(50)
    lsi.save(lsi_save_path) 

#建立LSI Model
def build_lsi_model(corpus,dictionary,tfidf_save_path,lsi_save_path):    
    corpus_tfidf = tfidf_model(corpus,tfidf_save_path)
    lsi_model(corpus_tfidf,dictionary,lsi_save_path)

#讀取LSI Model
def load_lsi_model(lsi_save_path):
    lsi = models.LsiModel.load(lsi_save_path)
    return lsi

#產生LDA Model
def lda_model(corpus,dictionary,lda_save_path,num_topics=300):
    model = models.LdaModel(corpus, id2word=dictionary, num_topics=num_topics)
    model.save(lda_save_path) 
    
#讀取LDA Model
def load_lad_model(lda_save_path):
    lda = models.LdaModel.load(lda_save_path)
    return lda

#建立RP Model
def build_rp_model(corpus,dictionary,tfidf_save_path,rp_save_path):
    tfidfmodel = load_tfidf_model(tfidf_save_path)
    corpus_tfidf = tfidfmodel[corpus]
    rp = models.RpModel(corpus_tfidf, num_topics=500)  
    rp.save(rp_save_path)

#讀取RP Model    
def load_rp_model(rp_save_path):
    return models.RpModel.load(rp_save_path)

def main(corpora_path,dicsavepath,tfidf_save_path,lsi_save_path,lda_save_path,rp_save_path):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    dictionary = corpora.Dictionary.load(dicsavepath)
    corpus = load_corpora(corpora_path)
    # build_rp_model(corpus,dictionary,tfidf_save_path,rp_save_path)
    build_lsi_model(corpus,dictionary,tfidf_save_path,lsi_save_path)
#     lda_model(corpus,dictionary,lda_save_path)

if __name__ == '__main__':
    corpora_path = 'data/qa_bot.mm'
    dicsavepath = 'data/qa_bot.dict'
    lsi_save_path = 'data/qa_bot.lsi'
    lda_save_path = 'data/qa_bot.lda'
    tfidf_save_path = 'data/qa_bot.tfidf'
    rp_save_path = 'data/qa_bot.rp'
    main(corpora_path,dicsavepath,tfidf_save_path,lsi_save_path,lda_save_path,rp_save_path)
        