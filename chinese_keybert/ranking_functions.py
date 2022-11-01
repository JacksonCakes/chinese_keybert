import numpy as np
import itertools
from sklearn.metrics.pairwise import cosine_similarity

def max_sum_sim(doc_embedding, candidate_embeddings, candidates, top_k, nr_candidates):
    """Calculate Max Sum Distance for extraction of keywords
    We take the nr_candidates most similar words/phrases to the document.
    Then, we take all top_k combinations from the nr_candidates and
    extract the combination that are the least similar to each other
    by cosine similarity.
    This is O(n^2) and therefore not advised if you use a large `top_k`
    Arguments:
        doc_embedding: The document embeddings
        candidate_embeddings: The embeddings of the selected candidate keywords/phrases
        candidates: The selected candidate keywords/keyphrases
        top_k: The number of keywords/keyhprases to return
        nr_candidates: The number of candidates to consider
    Returns:
         List[str]: The selected keywords/keyphrases
    """
    if nr_candidates < top_k:
        raise Exception(
            "Make sure that the number of candidates exceeds the number "
            "of keywords to return."
        )
    elif top_k > len(candidates):
        return []

    # Calculate distances and extract keywords
    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    distances_candidates = cosine_similarity(candidate_embeddings, 
                                            candidate_embeddings)

    # Get top_k words as candidates based on cosine similarity
    words_idx = list(distances.argsort()[0][-nr_candidates:])
    words_vals = [candidates[index] for index in words_idx]
    distances_candidates = distances_candidates[np.ix_(words_idx, words_idx)]

    # Calculate the combination of words that are the least similar to each other
    min_sim = np.inf
    candidate = None
    for combination in itertools.combinations(range(len(words_idx)), top_k):
        sim = sum([distances_candidates[i][j] for i in combination for j in combination if i != j])
        if sim < min_sim:
            candidate = combination
            min_sim = sim

    return [words_vals[idx] for idx in candidate]


def mmr(doc_embedding, candidate_embeddings, words, top_k, diversity):
    """Calculate Maximal Marginal Relevance (MMR)
    between candidate keywords and the document.

    MMR considers the similarity of keywords/keyphrases with the
    document, along with the similarity of already selected
    keywords and keyphrases. This results in a selection of keywords
    that maximize their within diversity with respect to the document.
    Arguments:
        doc_embedding: The document embeddings
        word_embeddings: The embeddings of the selected candidate keywords/phrases
        words: The selected candidate keywords/keyphrases
        top_k: The number of keywords/keyhprases to return
        diversity: How diverse the select keywords/keyphrases are.
                   Values between 0 and 1 with 0 being not diverse at all
                   and 1 being most diverse.
    Returns:
         List[str]: The selected keywords/keyphrases with their distances
    """
    # Extract similarity within words, and between words and the document
    word_doc_similarity = cosine_similarity(candidate_embeddings, doc_embedding) 
    word_similarity = cosine_similarity(candidate_embeddings) 

    # Initialize candidates and already choose best keyword/keyphras
    keywords_idx = [np.argmax(word_doc_similarity)]  
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(top_k - 1):
        # Extract similarities within candidates and
        # between candidates and selected keywords/phrases
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # Calculate MMR
        mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # Update keywords & candidates
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]