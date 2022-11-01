from chinese_keybert import Chinese_Extractor
import pytest
from .utils import get_test_data

kw_extractor = Chinese_Extractor()
docs1,docs2 = get_test_data()

@pytest.mark.parametrize("n_keywords", [i for i in range(1,5)])
@pytest.mark.parametrize("methods",['max_sum','mmr'])
def test_single_str_doc(n_keywords, methods):
    """Test whether the keywords are correctly extracted"""

    keywords = kw_extractor.generate_keywords(
        docs1[0],
        top_k=n_keywords,
        rank_methods=methods
    )
    assert isinstance(keywords, list)

    assert len(keywords[0]) == n_keywords

@pytest.mark.parametrize("n_keywords", [i for i in range(1,5)])
@pytest.mark.parametrize("methods",['max_sum','mmr'])
def test_single_list_doc(n_keywords, methods):
    """Test whether the keywords are correctly extracted"""

    keywords = kw_extractor.generate_keywords(
        docs1,
        top_k=n_keywords,
        rank_methods=methods
    )
    assert isinstance(keywords, list)

    assert len(keywords[0]) == n_keywords

@pytest.mark.parametrize("n_keywords", [i for i in range(1,5)])
@pytest.mark.parametrize("methods",['max_sum','mmr'])
def test_single_multiple_doc(n_keywords, methods):
    """Test whether the keywords are correctly extracted"""

    keywords = kw_extractor.generate_keywords(
        [docs1[0],docs2[0]],
        top_k=n_keywords,
        rank_methods=methods
    )
    assert isinstance(keywords, list)
    assert len(keywords) == 2
    assert len(keywords[0]) == n_keywords

def test_empty_doc():
    """Test empty document"""
    docs = ""
    result = kw_extractor.generate_keywords(docs)

    assert result == []