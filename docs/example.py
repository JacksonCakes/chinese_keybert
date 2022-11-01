from chinese_keybert import Chinese_Extractor

kw_extractor = Chinese_Extractor()

text = ['''
    深度學習是機器學習的分支， 透過模仿人類大腦及受其啟發的演算法改進、演化自己。 這些演算法的官方名稱是人工神經網絡。
程式員可以用文字、圖片及聲音等大量不同形式且複雜的無標籤數據來「訓練」這些神經網絡。然後這些模型便會不時「從經驗中學習」，
最終達至前所未有的準確度，甚至超越人類所及。雖然難以置信，但深度學習模式這個概念其實可追溯至1943年。
當時就已有兩名科學家利用數學及演算法，創建出模仿人類大腦的多層神經網絡。
    ''']
    
result = kw_extractor.generate_keywords(text,top_k=5,rank_methods="mmr",diversity=0.6)