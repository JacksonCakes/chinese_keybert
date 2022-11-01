# Chinese_keyBERT
Chinese_keyBERT is a minimal Chinese keywords extraction library that leverage the contextual embeddings generated from BERT models to extract relevant keywords from the given texts.

## Installation
```
pip install chinese_keybert
```

## Get started
```
from chinese_keybert import Chinese_Extractor

kw_extractor = Chinese_Extractor()

text = ['''
    深度學習是機器學習的分支， 透過模仿人類大腦及受其啟發的演算法改進、演化自己。 這些演算法的官方名稱是人工神經網絡。
程式員可以用文字、圖片及聲音等大量不同形式且複雜的無標籤數據來「訓練」這些神經網絡。然後這些模型便會不時「從經驗中學習」，
最終達至前所未有的準確度，甚至超越人類所及。雖然難以置信，但深度學習模式這個概念其實可追溯至1943年。
當時就已有兩名科學家利用數學及演算法，創建出模仿人類大腦的多層神經網絡。
    ''']
    
result = kw_extractor.generate_keywords(text,top_k=5,rank_methods="mmr",diversity=0.6)
>> [['深度', '演算法', '前所未有', '模型', '網絡']]
```

## How it works
The core idea behind chinese_keyBERT is to utilize a word segmentation models to segments a piece of text into smaller n-grams and filter the n-grams according to the defined part-of-speech (as some pos are not suitable to be used as a keyword). Then, an embedding model (eg. BERT) is used to encode the text and filtered n_grams into embeddings and using some ranking methods (eg. maximun sum/maximun marginal relevance) to compute the cosine distances betweens the text and n-grams embeddings and rank the keywords according to the scores.

## To-do
- [ ] Documentations
- [ ] Vectorization operations to speed-up processing of multiple documents
- [ ] Add support for other word segmentation, part-of-speech and embeddings model

## Credit
Chinese_keyBERT was largely inspired by [KeyBERT](https://github.com/MaartenGr/KeyBERT), a minimal library for embedding based keywords extractions. Besides, Chinese_keyBERT is also heavily relies on Chinese word segmentation and POS library from [CKIP](https://github.com/ckiplab/ckip-transformers) as well as [sentence-transformer](https://github.com/UKPLab/sentence-transformers) for generating quality embeddings. 
