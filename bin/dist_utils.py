# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: utils for distance computation

"""

import sys
import warnings
warnings.filterwarnings("ignore")

try:
    import lzma
    import Levenshtein
except:
    pass
import numpy as np
from difflib import SequenceMatcher
from sklearn.metrics.pairwise import cosine_similarity
import config

import np_utils
sys.path.append("..")


def _edit_dist(str1, str2):
    try:
        # very fast
        # http://stackoverflow.com/questions/14260126/how-python-levenshtein-ratio-is-computed
        # d = Levenshtein.ratio(str1, str2)
        d = Levenshtein.distance(str1, str2)/float(max(len(str1),len(str2)))
    except:
        # https://docs.python.org/2/library/difflib.html
        d = 1. - SequenceMatcher(lambda x: x==" ", str1, str2).ratio()
    return d


def _is_str_match(str1, str2, threshold=1.0):
    assert threshold >= 0.0 and threshold <= 1.0, "Wrong threshold."
    if float(threshold) == 1.0:
        return str1 == str2
    else:
        return (1. - _edit_dist(str1, str2)) >= threshold


def _longest_match_size(str1, str2):
    sq = SequenceMatcher(lambda x: x==" ", str1, str2)
    match = sq.find_longest_match(0, len(str1), 0, len(str2))
    return match.size


def _longest_match_ratio(str1, str2):
    sq = SequenceMatcher(lambda x: x==" ", str1, str2)
    match = sq.find_longest_match(0, len(str1), 0, len(str2))
    return np_utils._try_divide(match.size, min(len(str1), len(str2)))


def _compression_dist(x, y, l_x=None, l_y=None):
    if x == y:
        return 0
    x_b = x.encode('utf-8')
    y_b = y.encode('utf-8')
    if l_x is None:
        l_x = len(lzma.compress(x_b))
        l_y = len(lzma.compress(y_b))
    l_xy = len(lzma.compress(x_b+y_b))
    l_yx = len(lzma.compress(y_b+x_b))
    dist = np_utils._try_divide(min(l_xy,l_yx)-min(l_x,l_y), max(l_x,l_y))
    return dist


def _cosine_sim(vec1, vec2):
    try:
        s = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]
    except:
        try:
            s = cosine_similarity(vec1, vec2)[0][0]
        except:
            s = config.MISSING_VALUE_NUMERIC
    return s


def _vdiff(vec1, vec2):
    return vec1 - vec2


def _rmse(vec1, vec2):
    vdiff = vec1 - vec2
    rmse = np.sqrt(np.mean(vdiff**2))
    return rmse


def _KL(dist1, dist2):
    "Kullback-Leibler Divergence"
    return np.sum(dist1 * np.log(dist1/dist2), axis=1)


def _jaccard_coef(A, B):
    if not isinstance(A, set):
        A = set(A)
    if not isinstance(B, set):
        B = set(B)
    return np_utils._try_divide(float(len(A.intersection(B))), len(A.union(B)))


def _dice_dist(A, B):
    if not isinstance(A, set):
        A = set(A)
    if not isinstance(B, set):
        B = set(B)
    return np_utils._try_divide(2.*float(len(A.intersection(B))), (len(A) + len(B)))
