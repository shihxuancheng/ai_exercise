# AI Exercise - QA Robot (SEC Data)
透過[自然語處理(Natural Language Processing, NLP)](https://zh.wikipedia.org/wiki/%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86)方式，分析及拆解使用者提出的問題，由SEC QA資料蒐尋出最近似的答案

### NLP應用範圍

1.  文本朗讀（Text to speech）/ 語音合成（Speech synthesis）
2.  語音識別（Speech recognition）
3.  **`自動分詞（word segmentation）`**
4.  詞性標註（Part-of-speech tagging）
5.  句法分析（Parsing）
6.  自然語言生成（Natural language generation）
7.  **`文本分類（Text categorization）`**
8.  **`信息檢索（Information retrieval）`**
9.  信息抽取（Information extraction）
10. 文字校對（Text-proofing）
11. 問答系統（Question answering）
12. 機器翻譯（Machine translation）
13. 自動摘要（Automatic summarization）
14. 文字蘊涵（Textual entailment）

## 基本概念

### 1. 將詞句轉換為向量

* 基於統計原理，如 TFIDF、LSI、LDA...，以詞袋 (Bag Of Word, BOW)形式表現
* 基於深度學習，如word2vec、doc2vec， 以詞向量 (Word Embedding)形式表現

舉例

    句子1:  我喜歡AI，不喜歡BI
    句子2:  我不喜歡AI，也不喜歡BI

第一步 分詞 (jieba)

    句子1:  我 / 喜歡 / AI / 不 / 喜歡 / BI
    句子2:  我 / 不 / 喜歡 / AI / 也 / 不 ／ 喜歡 / BI

第二步 列出所有詞

    我，喜歡，不，AI，BI，也

第三步 分別計算TF

||我|喜歡|不|AI|BI|也|
|-|--|---|--|--|--|--|
|句子1|1|2|1|1|1|0|
|句子2|1|2|2|1|1|1|

句子1: [1, 2, 1, 1, 1, 0], 句子2: [1, 2, 2, 1, 1, 1]


***

### 2. TF - IDF 計算法 ( term frequency–inverse document frequency )
tf - idf 是一種統計方法，此原理為評估一個字詞對於一個檔案集，或一個語料庫中的其中一份檔案的重要程度，這個概念十分重要。

#### TF (term frequency) - 詞頻，代表某個詞在文章中出現的頻率
#### IDF (inverse document frequency) - 逆文件頻率， 衡量某個詞的重要性

![test](https://raw.githubusercontent.com/shihxuancheng/ai_exercise/master/resources/images/img-2.png)

### 3. Cosine Similarity (餘弦相似性計算)
餘絃相似度（cosine similarity）是資訊檢索中常用的相似度計算方式，可用來計算文件之間的相似度，
也可以計算詞彙之間的相似度，更可以計算查詢字串與文件之間的相似度。

![cosine_similar-1](https://raw.githubusercontent.com/shihxuancheng/ai_exercise/master/resources/images/img-11.png)
![cosine_simi-2](https://raw.githubusercontent.com/shihxuancheng/ai_exercise/master/resources/images/img-12.png)


## Solutions
### `1. TF-IDF + Cosine similar`
#### 基本思路
1.  資料預處理: 語料庫中所有問題進行中文分詞，去除重複、停用字及低詞頻雜訊
2.  所有分詞結果整理成一個集合並計算IDF
2.  將每個問題轉換為向量(bow)並計算TFIDF
3.  將user提出的問題轉換為向量(bow)並計算TFIDF
4.  計算兩者餘弦相似度(越接近1表越相似)，回傳最相似的結果

*** 

### `2. LSI/LDA Model`
#### 基本思路
1.  資料預處理: 語料庫中所有問題進行中文分詞，去除重複、停用字及低詞頻雜訊
2.  分詞結果整理成一個集合並轉換為字典檔 (word -> id)
3.  透過字典檔將語料庫轉換為向量格式
4.  將語料庫轉換為**TFIDF model**
5.  透過**TFIDF model**建立**LSI Model**並指定**topic**數量
6.  根據**topics**計算建立索引
7.  將user提出的問題轉換為向量(bow)並透過建立好的lsi model計算出最相似的答案

### 何謂LSI 模型 
#### [LSI (Latent Semantic Indexing) - 潛在語義索引](https://raymondyangsite.wordpress.com/2017/05/03/110/)
是利用 [SVD ( Singular Value Decomposition )](https://www.zhihu.com/question/22237507)把文件從高維空間投影到低維空間(topics)，在這個空間內進行文本相似性的比較。與詞組之間語意上的關聯相比， LSI 更關注的是詞組之間「隱含」的關聯


## 執行環境
### Environment &  Modules
* python 3
* Modules: **gensim** / **jieba** / **wordcloud** / **pandas** / **matplotlib**

    ``` pip install -r require_pkg.txt ```
* Run `Demo.ipynb` with jupyter notebook

### Docker
```yml
    docker run -p 8888:8888 shihxuancheng/ai_exercise
```
<!-- ### NLP
自然語言處理最難的就是語言的多樣性 -->