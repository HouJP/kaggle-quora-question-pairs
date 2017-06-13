# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: utils for ngram

"""


def _unigrams(words):
    """
        Input: a list of words, e.g., ["I", "am", "Denny"]
        Output: a list of unigram
    """
    assert type(words) == list
    return words


def _bigrams(words, join_string, skip=0):
    """
       Input: a list of words, e.g., ["I", "am", "Denny"]
       Output: a list of bigram, e.g., ["I_am", "am_Denny"]
       I use _ as join_string for this example.
    """
    assert type(words) == list
    L = len(words)
    if L > 1:
        lst = []
        for i in range(L-1):
            for k in range(1,skip+2):
                if i+k < L:
                    lst.append( join_string.join([words[i], words[i+k]]) )
    else:
        # set it as unigram
        lst = _unigrams(words)
    return lst


def _trigrams(words, join_string, skip=0):
    """
       Input: a list of words, e.g., ["I", "am", "Denny"]
       Output: a list of trigram, e.g., ["I_am_Denny"]
       I use _ as join_string for this example.
    """
    assert type(words) == list
    L = len(words)
    if L > 2:
        lst = []
        for i in range(L-2):
            for k1 in range(1,skip+2):
                for k2 in range(1,skip+2):
                    if i+k1 < L and i+k1+k2 < L:
                        lst.append( join_string.join([words[i], words[i+k1], words[i+k1+k2]]) )
    else:
        # set it as bigram
        lst = _bigrams(words, join_string, skip)
    return lst


def _fourgrams(words, join_string):
    """
        Input: a list of words, e.g., ["I", "am", "Denny", "boy"]
        Output: a list of trigram, e.g., ["I_am_Denny_boy"]
        I use _ as join_string for this example.
    """
    assert type(words) == list
    L = len(words)
    if L > 3:
        lst = []
        for i in xrange(L-3):
            lst.append( join_string.join([words[i], words[i+1], words[i+2], words[i+3]]) )
    else:
        # set it as trigram
        lst = _trigrams(words, join_string)
    return lst


def _uniterms(words):
    return _unigrams(words)


def _biterms(words, join_string):
    """
        Input: a list of words, e.g., ["I", "am", "Denny", "boy"]
        Output: a list of biterm, e.g., ["I_am", "I_Denny", "I_boy", "am_Denny", "am_boy", "Denny_boy"]
        I use _ as join_string for this example.
    """
    assert type(words) == list
    L = len(words)
    if L > 1:
        lst = []
        for i in range(L-1):
            for j in range(i+1,L):
                lst.append( join_string.join([words[i], words[j]]) )
    else:
        # set it as uniterm
        lst = _uniterms(words)
    return lst


def _triterms(words, join_string):
    """
        Input: a list of words, e.g., ["I", "am", "Denny", "boy"]
        Output: a list of triterm, e.g., ["I_am_Denny", "I_am_boy", "I_Denny_boy", "am_Denny_boy"]
        I use _ as join_string for this example.
    """
    assert type(words) == list
    L = len(words)
    if L > 2:
        lst = []
        for i in xrange(L-2):
            for j in xrange(i+1,L-1):
                for k in xrange(j+1,L):
                    lst.append( join_string.join([words[i], words[j], words[k]]) )
    else:
        # set it as biterm
        lst = _biterms(words, join_string)
    return lst


def _fourterms(words, join_string):
    """
        Input: a list of words, e.g., ["I", "am", "Denny", "boy", "ha"]
        Output: a list of fourterm, e.g., ["I_am_Denny_boy", "I_am_Denny_ha", "I_am_boy_ha", "I_Denny_boy_ha", "am_Denny_boy_ha"]
        I use _ as join_string for this example.
    """
    assert type(words) == list
    L = len(words)
    if L > 3:
        lst = []
        for i in xrange(L-3):
            for j in xrange(i+1,L-2):
                for k in xrange(j+1,L-1):
                    for l in xrange(k+1,L):
                        lst.append( join_string.join([words[i], words[j], words[k], words[l]]) )
    else:
        # set it as triterm
        lst = _triterms(words, join_string)
    return lst


_ngram_str_map = {
    1: "Unigram",
    2: "Bigram",
    3: "Trigram",
    4: "Fourgram",
    5: "Fivegram",
    12: "UBgram",
    123: "UBTgram",
}


def _ngrams(words, ngram, join_string=" "):
    """wrapper for ngram"""
    if ngram == 1:
        return _unigrams(words)
    elif ngram == 2:
        return _bigrams(words, join_string)
    elif ngram == 3:
        return _trigrams(words, join_string)
    elif ngram == 4:
        return _fourgrams(words, join_string)
    elif ngram == 12:
        unigram = _unigrams(words)
        bigram = [x for x in _bigrams(words, join_string) if len(x.split(join_string)) == 2]
        return unigram + bigram
    elif ngram == 123:
        unigram = _unigrams(words)
        bigram = [x for x in _bigrams(words, join_string) if len(x.split(join_string)) == 2]
        trigram = [x for x in _trigrams(words, join_string) if len(x.split(join_string)) == 3]
        return unigram + bigram + trigram


_nterm_str_map = {
    1: "Uniterm",
    2: "Biterm",
    3: "Triterm",
    4: "Fourterm",
    5: "Fiveterm",
}


def _nterms(words, nterm, join_string=" "):
    """wrapper for nterm"""
    if nterm == 1:
        return _uniterms(words)
    elif nterm == 2:
        return _biterms(words, join_string)
    elif nterm == 3:
        return _triterms(words, join_string)
    elif nterm == 4:
        return _fourterms(words, join_string)


if __name__ == "__main__":

    text = "I am Denny boy ha"
    words = text.split(" ")

    assert _ngrams(words, 1) == ["I", "am", "Denny", "boy", "ha"]
    assert _ngrams(words, 2) == ["I am", "am Denny", "Denny boy", "boy ha"]
    assert _ngrams(words, 3) == ["I am Denny", "am Denny boy", "Denny boy ha"]
    assert _ngrams(words, 4) == ["I am Denny boy", "am Denny boy ha"]

    assert _nterms(words, 1) == ["I", "am", "Denny", "boy", "ha"]
    assert _nterms(words, 2) == ["I am", "I Denny", "I boy", "I ha", "am Denny", "am boy", "am ha", "Denny boy", "Denny ha", "boy ha"]
    assert _nterms(words, 3) == ["I am Denny", "I am boy", "I am ha", "I Denny boy", "I Denny ha", "I boy ha", "am Denny boy", "am Denny ha", "am boy ha", "Denny boy ha"]
    assert _nterms(words, 4) == ["I am Denny boy", "I am Denny ha", "I am boy ha", "I Denny boy ha", "am Denny boy ha"]
