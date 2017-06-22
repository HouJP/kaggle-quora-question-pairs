#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/6/21 00:43
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


import csv
import json
import math

import numpy as np
import pandas as pd
from numpy import linalg

from bin.featwheel.utils import LogUtil
from ..featwheel.extractor import Extractor


class BTM(Extractor):
    @staticmethod
    def load_questions_btm_vectors(qid_fp, qf_fp):
        fqid = open(qid_fp, 'r')
        qids = fqid.readlines()
        fqid.close()

        fqf = open(qf_fp, 'r')
        qfs = fqf.readlines()
        fqf.close()

        assert len(qids) == len(qfs), "len(qid) != len(question)"

        btm_vecs = {}
        for index in range(len(qids)):
            btm_vecs[str(qids[index]).strip()] = qfs[index].strip().split()
        return btm_vecs

    def __init__(self, config_fp, qid_fp, qf_fp):
        Extractor.__init__(self, config_fp)
        self.btm_vectors = BTM.load_questions_btm_vectors(qid_fp, qf_fp)

    def extract_row(self, row):
        q1_id = str(row['qid1'])
        q2_id = str(row['qid2'])
        q1_features = self.btm_vectors[q1_id]
        q2_features = self.btm_vectors[q2_id]
        return q1_features + q2_features

    def get_feature_num(self):
        return len(self.btm_vectors.values()[0]) * 2


class WordEmbedding(Extractor):
    @staticmethod
    def generate_idf(data_fp):
        data = csv.reader(data_fp)
        idf = {}
        for index, row in data.iterrows():
            words = str(row['question']).strip().split() if WordEmbedding.to_lower else str(
                row['question']).lower().strip().split()
            for word in words:
                idf[word] = idf.get(word, 0) + 1
        num_docs = len(data)
        for word in idf:
            idf[word] = math.log(num_docs / (idf[word] + 1.)) / math.log(2.)
        LogUtil.log("INFO", "IDF calculation done, len(idf)=%d" % len(idf))
        return idf

    @staticmethod
    def load_word_embedding(we_fp):
        we_dic = {}
        f = open(we_fp, 'r')
        for line in f:
            subs = line.strip().split(None, 1)
            if 2 > len(subs):
                continue
            else:
                word = subs[0]
                vec = subs[1]
            we_dic[word] = np.array([float(s) for s in vec.split()])
        f.close()
        return we_dic


class WordEmbeddingAveDis(Extractor):

    def __init__(self, config_fp, word_embedding_fp, to_lower=True):
        Extractor.__init__(self, config_fp)
        self.we_dic = WordEmbedding.load_word_embedding(word_embedding_fp)
        self.we_len = len(self.we_dic.values()[0])
        self.to_lower = to_lower

    def extract_row(self, row):
        q1_words = str(row['question1']).strip().split() if self.to_lower else str(
            row['question1']).lower().strip().split()
        q2_words = str(row['question2']).strip().split() if self.to_lower else str(
            row['question2']).lower().strip().split()
        q1_vec = np.array(self.we_len * [0.])
        q2_vec = np.array(self.we_len * [0.])

        for word in q1_words:
            if word in self.we_dic:
                q1_vec = q1_vec + self.we_dic[word]
        for word in q2_words:
            if word in self.we_dic:
                q2_vec = q2_vec + self.we_dic[word]

        cos_sim = 0.
        q1_vec = np.mat(q1_vec)
        q2_vec = np.mat(q2_vec)
        factor = linalg.norm(q1_vec) * linalg.norm(q2_vec)
        if 1e-6 < factor:
            cos_sim = float(q1_vec * q2_vec.T) / factor
        return [cos_sim]

    def get_feature_num(self):
        return 1


class WordEmbeddingTFIDFAveDis(WordEmbeddingAveDis):

    def __init__(self, config_fp, word_embedding_fp, qid2q_fp, to_lower=True):
        WordEmbeddingAveDis.__init__(self, config_fp, word_embedding_fp, to_lower)
        self.idf = WordEmbedding.generate_idf(qid2q_fp)

    def extract_row(self, row):
        q1_words = str(row['question1']).strip().split() if 'True' == WordEmbedding.to_lower else str(
            row['question1']).lower().strip().split()
        q2_words = str(row['question2']).strip().split() if 'True' == WordEmbedding.to_lower else str(
            row['question2']).lower().strip().split()

        q1_vec = np.array(self.we_len * [0.])
        q2_vec = np.array(self.we_len * [0.])
        q1_words_cnt = {}
        q2_words_cnt = {}
        for word in q1_words:
            q1_words_cnt[word] = q1_words_cnt.get(word, 0.) + 1.
        for word in q2_words:
            q2_words_cnt[word] = q2_words_cnt.get(word, 0.) + 1.

        for word in q1_words_cnt:
            if word in self.we_dic:
                q1_vec += self.idf.get(word, 0.) * q1_words_cnt[word] * WordEmbedding.we_dict[word]
        for word in q2_words_cnt:
            if word in self.we_dic:
                q2_vec += self.idf.get(word, 0.) * q2_words_cnt[word] * WordEmbedding.we_dict[word]

        cos_sim = 0.
        q1_vec = np.mat(q1_vec)
        q2_vec = np.mat(q2_vec)
        factor = linalg.norm(q1_vec) * linalg.norm(q2_vec)
        if 1e-6 < factor:
            cos_sim = float(q1_vec * q2_vec.T) / factor
        return [cos_sim]


class WordEmbeddingAveVec(WordEmbeddingAveDis):
    def extract_row(self, row):
        q1_words = str(row['question1']).strip().split() if self.to_lower else str(
            row['question1']).lower().strip().split()
        q2_words = str(row['question2']).strip().split() if self.to_lower else str(
            row['question2']).lower().strip().split()

        q1_vec = np.array(self.we_len * [0.])
        q2_vec = np.array(self.we_len * [0.])

        for word in q1_words:
            if word in self.we_dic:
                q1_vec += self.we_dic[word]
        for word in q2_words:
            if word in self.we_dic:
                q2_vec += self.we_dic[word]

        return list(q1_vec) + list(q2_vec)

    def get_feature_num(self):
        return self.we_len * 2


class WordEmbeddingTFIDFAveVec(WordEmbeddingTFIDFAveDis):
    def extract_row(self, row):
        q1_words = str(row['question1']).strip().split() if self.to_lower else str(
            row['question1']).lower().strip().split()
        q2_words = str(row['question2']).strip().split() if self.to_lower else str(
            row['question2']).lower().strip().split()

        q1_vec = np.array(self.we_len * [0.])
        q2_vec = np.array(self.we_len * [0.])

        q1_words_cnt = {}
        q2_words_cnt = {}
        for word in q1_words:
            q1_words_cnt[word] = q1_words_cnt.get(word, 0.) + 1.
        for word in q2_words:
            q2_words_cnt[word] = q2_words_cnt.get(word, 0.) + 1.

        for word in q1_words_cnt:
            if word in self.we_dic:
                q1_vec += self.idf.get(word, 0.) * q1_words_cnt[word] * self.we_dic[word]
        for word in q2_words_cnt:
            if word in self.we_dic:
                q2_vec += self.idf.get(word, 0.) * q2_words_cnt[word] * self.we_dic[word]

        return list(q1_vec) + list(q2_vec)

    def get_feature_num(self):
        return self.we_len * 2


class POSTagCount(Extractor):

    @staticmethod
    def load_postag(config, data_set_name):
        # load data set from disk
        data = pd.read_csv('%s/%s.csv' % (config.get('DEFAULT', 'source_pt'), data_set_name)).fillna(value="")
        postag = {}
        for index, row in data:
            q1_postag = json.loads(row['question1_postag'])
            for sentence in q1_postag:
                for kv in sentence:
                    postag.setdefault(kv[1], len(postag))

            q2_postag = json.loads(row['question2_postag'])
            for sentence in q2_postag:
                for kv in sentence:
                    postag.setdefault(kv[1], len(postag))
        return postag

    def __init__(self, config_fp, data_set_name):
        Extractor.__init__(self, config_fp)
        self.postag = POSTagCount.load_postag(self.config, data_set_name)

    def extract_row(self, row):
        q1_vec = len(self.postag) * [0]
        q1_postag = json.loads(row['question1_postag'])
        for s in q1_postag:
            for kv in s:
                postag_id = self.postag[kv[1]]
                q1_vec[postag_id] += 1
        q2_vec = len(self.postag) * [0]
        q2_postag = json.loads(row['question2_postag'])
        for s in q2_postag:
            for kv in s:
                postag_id = self.postag[kv[1]]
                q2_vec[postag_id] += 1

        q1_vec = np.array(q1_vec)
        q2_vec = np.array(q2_vec)
        sum_vec = q1_vec + q2_vec
        sub_vec = abs(q1_vec - q2_vec)
        dot_vec = q1_vec.dot(q2_vec)
        q1_len = np.sqrt(q1_vec.dot(q1_vec))
        q2_len = np.sqrt(q2_vec.dot(q2_vec))
        cos_sim = 0.
        if q1_len * q2_len > 1e-6:
            cos_sim = dot_vec / q1_len / q2_len
        return list(q1_vec) + list(q2_vec) + list(sum_vec) + list(sub_vec) + [dot_vec, q1_len, q2_len, cos_sim]

    def get_feature_num(self):
        return len(self.postag) * 4 + 4

