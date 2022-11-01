from chinese_keybert import Chinese_Extractor

kw_extractor = Chinese_Extractor()
text = [
   '''
渾水創始人：七月開始調查貝殼，因為“好得難以置信” 2021年12月16日，做空機構渾水在社交媒體上公開表示，正在做空美股上市公司貝殼...
'''
]
result = kw_extractor.generate_keywords(text,rank_methods="mmr")
print(result)