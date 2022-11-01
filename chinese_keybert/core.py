from ckip_transformers.nlp import CkipWordSegmenter,CkipPosTagger
from sentence_transformers import SentenceTransformer
import numpy as np
from .stopwords import read_stopwords_list
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Union
from .ranking_functions import max_sum_sim,mmr

class Chinese_Extractor():
    def __init__(
        self,
        ws_model: str ="bert-base",
        pos_model: str = "bert-base",
        embeddings_model: str = "distiluse-base-multilingual-cased-v1",
        device: int = -1):
        """
        Arguments:
            ws_model: Word segmentation model (Currently supported models are from https://ckip-transformers.readthedocs.io/en/stable/)
            pos_model: Model to perform part of speech tagging (Currently supported models are from https://ckip-transformers.readthedocs.io/en/stable/)
            embeddings_model: Model to encode docs into embeddings representation (Currently supported models are from https://www.sbert.net/docs/pretrained_models.html)
            device: Whether to use CPU -1 (default) or GPU (0) for processing
        """
        self.ws_model = CkipWordSegmenter(model=ws_model,device=device)
        self.pos_model = CkipPosTagger(model=pos_model,device=device)
        self.embeddings_model = SentenceTransformer(embeddings_model)
        self.stopwords = read_stopwords_list()

    def generate_keywords(
        self,
        docs,
        top_k: int = 5,
        rank_methods: str = "max_sum",
        nr_candidates: int = 20,
        diversity: float = 0.5,
        use_delim: bool = True,
        batch_size: int = 256,
        max_length: int = 128,
        splitter: str = "\n\n",
        pos_list: List = ['Na','VH','Nc','VC']) -> List[str]:
        """
        Arguments:
            docs: The documents to perform keywords extraction.
            top_k: The number of keywords to be extract.
            rank_methods: Methods to select and diversify the keywords [max_sum,mmr].
            nr_candidates: Initial number of candidates, will select top_k keywords from nr_candidates, 
                           only applicable when methods is max_sum.
            diversity: The diversity between each keywords, range from 0-1, only applicable when methods is mmr.
            use_delim: If true, segmentation model will automatically use predefined symbol to segment the text.
            batch_size: Number of batch processing for more efficient processing (applicable to segmentation model).
            max_length: Max length of sentence to process for more efficient processing (applicable to segmentation model).
            splitter: Split large doc into smaller chunk according to input splitter for more accurate result.
            pos_list: List of pos tag to select keywords from.
        Returns:
            final_keywords = list of keywords extracted

        """
        if not docs:
            return []
            
        if isinstance(docs,str):
            docs = [docs]

        ws_list  = self.ws_model(docs,use_delim= use_delim,batch_size=batch_size, max_length=max_length)
        final_ws = []
        pos_dict_list = []      
        docs_candidates = []
        # Find all unique keywords candidates exclude stopwords
        for ws in ws_list:
            unique_ws = set(ws)
            ws_no_sw = [word for word in unique_ws if not word in self.stopwords]
            final_ws.append(ws_no_sw)
        pos = self.pos_model(final_ws)

        for doc, sentence_ws, sentence_pos in zip(docs,final_ws, pos):
            pos_dict_list.append(dict(self.pack_ws_pos_sentence(sentence_ws, sentence_pos)))
        
        # Select only words present in the POS as keywords candidates
        for pos_ in pos_dict_list:
            candidates = []
            for keys,values in pos_.items():
                if keys in pos_list:
                    candidates+=values
            docs_candidates.append(candidates)

        # Split each doc according to splitter to small chunk of text for better result
        docs_split = [text.split(splitter) for text in docs]

        candidates_embeddings = [self.embeddings_model.encode(single_candidates)for single_candidates in docs_candidates]
        docs_embeddings = [self.embeddings_model.encode(docs)for docs in docs_split]
        
        docs_embeddings_avg = [np.expand_dims((np.sum(embeddings,axis=0)/embeddings.shape[0]),0) for embeddings in docs_embeddings]
        if rank_methods == "max_sum":
            final_keywords = [max_sum_sim(doc_embeddings,candidate_embeddings,doc_candidates,top_k,nr_candidates) for 
                            doc_embeddings,candidate_embeddings,doc_candidates in zip(docs_embeddings_avg,candidates_embeddings,docs_candidates)]
        else:
            final_keywords = [mmr(doc_embeddings,candidate_embeddings,doc_candidates,top_k,diversity) for 
                            doc_embeddings,candidate_embeddings,doc_candidates in zip(docs_embeddings_avg,candidates_embeddings,docs_candidates)]
        return final_keywords

    # Pack word segmentation and part-of-speech results into dictionary
    def pack_ws_pos_sentence(self,sentence_ws, sentence_pos):
        assert len(sentence_ws) == len(sentence_pos)
        dict_result = defaultdict(list)
        for word_ws, word_pos in zip(sentence_ws, sentence_pos):
            dict_result[word_pos].append(word_ws)
        return dict_result
        

