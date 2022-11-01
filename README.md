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
text = [
   '''
渾水創始人：七月開始調查貝殼，因為“好得難以置信” 2021年12月16日，做空機構渾水在社交媒體上公開表示，正在做空美股上市公司貝殼...
'''
]
result = kw_extractor.generate_keywords(text,top_k=5,rank_methods="mmr")
```

## How it works
The core idea behind chinese_keyBERT is to utilize a word segmentation models to segments a piece of text into smaller n-grams and filter the n-grams according to the defined part-of-speech (as some pos are not suitable to be used as a keyword). Then, an embedding model (eg. BERT) is used to encode the text and filtered n_grams into embeddings and using some ranking methods (eg. maximun sum/maximun marginal relevance) to compute the cosine distances betweens the text and n-grams embeddings and rank the keywords according to the scores.

## To-do
- [ ] Documentations
- [ ] Vectorization operations to speed-up processing of multiple documents
- [ ] Add support for other word segmentation, part-of-speech and embeddings model

## Credit
Chinese_keyBERT was largely inspired by [KeyBERT](https://github.com/MaartenGr/KeyBERT), a minimal library for embedding based keywords extractions. Besides, Chinese_keyBERT is also heavily relies on Chinese word segmentation and POS library from [CKIP](https://github.com/ckiplab/ckip-transformers) as well as [sentence-transformer](https://github.com/UKPLab/sentence-transformers) for generating quality embeddings. 
