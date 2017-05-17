#! /usr/bin/python
# -*- coding: utf-8 -*-

import ConfigParser
from nltk.corpus import stopwords
from feature import Feature
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import numpy as np
import math
import nltk
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import LogUtil
import json
import sys, getopt
from numpy import linalg
from preprocessor import Preprocessor
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
import networkx as nx
import csv
from scipy import sparse
import ngram_utils
import dist_utils
import np_utils
from postprocessor import PostProcessor
import config
from model import Model
import random

class WordMatchShare(object):
    """
    提取<Q1,Q2>特征: word_match_share
    """

    stops = set(stopwords.words("english"))

    @staticmethod
    def word_match_share(row):
        """
        针对一行抽取特征
        :param row: DataFrame中的一行数据
        :return: 特征值
        """
        q1words = {}
        q2words = {}
        for word in str(row['question1']).lower().split():
            if word not in WordMatchShare.stops:
                q1words[word] = 1
        for word in str(row['question2']).lower().split():
            if word not in WordMatchShare.stops:
                q2words[word] = 1
        if len(q1words) == 0 or len(q2words) == 0:
            # The computer-generated chaff includes a few questions that are nothing but stopwords
            return [0.]
        shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
        shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
        R = 1.0 * (len(shared_words_in_q1) + len(shared_words_in_q2)) / (len(q1words) + len(q2words))
        return [R]

    @staticmethod
    def run(train_df, test_df, feature_pt):
        """
        抽取train.csv test.csv特征
        :param train_df: train.csv数据
        :param test_df: test.csv数据
        :param feature_pt: 特征文件路径
        :return: None
        """
        train_features = train_df.apply(WordMatchShare.word_match_share, axis=1, raw=True)
        Feature.save_dataframe(train_features, feature_pt + '/word_match_share.train.smat')

        test_features = test_df.apply(WordMatchShare.word_match_share, axis=1, raw=True)
        Feature.save_dataframe(test_features, feature_pt + '/word_match_share.test.smat')

        word_match_share = train_features.apply(lambda x: x[0])
        plt.figure(figsize=(15, 5))
        plt.hist(word_match_share[train_df['is_duplicate'] == 0], bins=20, normed=True, label='Not Duplicate')
        plt.hist(word_match_share[train_df['is_duplicate'] == 1], bins=20, normed=True, alpha=0.7, label='Duplicate')
        plt.legend()
        plt.title('Label distribution over word_match_share', fontsize=15)
        plt.xlabel('word_match_share', fontsize=15)
        plt.show()

        return

    @staticmethod
    def demo():
        """
        使用样例
        :return: NONE
        """
        # 读取配置文件
        cf = ConfigParser.ConfigParser()
        cf.read("../conf/python.conf")

        # 加载train.csv文件
        train_data = pd.read_csv('%s/train.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")
        # 加载test.csv文件
        test_data = pd.read_csv('%s/test.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")
        # 特征文件路径
        feature_path = cf.get('DEFAULT', 'feature_question_pair_pt')
        # 提取特征
        WordMatchShare.run(train_data, test_data, feature_path)


class TFIDFWordMatchShare(object):
    """
    基于TF-IDF的词共享特征
    """

    stops = set(stopwords.words("english"))
    weights = {}

    @staticmethod
    def cal_weight(count, eps=10000, min_count=2):
        """
        根据词语出现次数计算权重
        :param count: 词语出现次数
        :param eps: 平滑常数
        :param min_count: 最少出现次数
        :return: 权重
        """
        if count < min_count:
            return 0.
        else:
            return 1. / (count + eps)

    @staticmethod
    def get_weights(data):
        """
        获取weight字典
        :param data: 数据集，一般是train.csv
        :return: None
        """
        qs_data = pd.Series(data['question1'].tolist() + data['question2'].tolist()).astype(str)
        words = (" ".join(qs_data)).lower().split()
        counts = Counter(words)
        TFIDFWordMatchShare.weights = {word: TFIDFWordMatchShare.cal_weight(count) for word, count in counts.items()}

    @staticmethod
    def tfidf_word_match_share(row):
        """
        针对一个<Q1,Q2>抽取特征
        :param row: 一个<Q1,Q2>实例
        :return: 特征值
        """
        q1words = {}
        q2words = {}
        for word in str(row['question1']).lower().split():
            if word not in TFIDFWordMatchShare.stops:
                q1words[word] = 1
        for word in str(row['question2']).lower().split():
            if word not in TFIDFWordMatchShare.stops:
                q2words[word] = 1
        if len(q1words) == 0 or len(q2words) == 0:
            # The computer-generated chaff includes a few questions that are nothing but stopwords
            return [0.]

        shared_weights = [TFIDFWordMatchShare.weights.get(w, 0) for w in q1words.keys() if w in q2words] + [
            TFIDFWordMatchShare.weights.get(w, 0) for w in q2words.keys() if w in q1words]
        total_weights = [TFIDFWordMatchShare.weights.get(w, 0) for w in q1words] + [
            TFIDFWordMatchShare.weights.get(w, 0)
            for w in q2words]
        if 1e-6 > np.sum(total_weights):
            return [0.]
        R = np.sum(shared_weights) / np.sum(total_weights)
        return [R]

    @staticmethod
    def run(train_df, test_df, feature_pt):
        """
        抽取train.csv和test.csv的<Q1,Q2>特征：tfidf_word_match_share
        :param train_df: train.csv
        :param test_df: test.csv
        :param feature_pt: 特征文件目录
        :return: None
        """
        # 获取weights信息
        TFIDFWordMatchShare.get_weights(train_df)
        # 抽取特征
        train_features = train_df.apply(TFIDFWordMatchShare.tfidf_word_match_share, axis=1, raw=True)
        Feature.save_dataframe(train_features, feature_pt + '/tfidf_word_match_share.train.smat')
        test_features = test_df.apply(TFIDFWordMatchShare.tfidf_word_match_share, axis=1, raw=True)
        Feature.save_dataframe(test_features, feature_pt + '/tfidf_word_match_share.test.smat')
        # 绘图
        tfidf_word_match_share = train_features.apply(lambda x: x[0])
        plt.figure(figsize=(15, 5))
        plt.hist(tfidf_word_match_share[train_df['is_duplicate'] == 0].fillna(0), bins=20, normed=True,
                 label='Not Duplicate')
        plt.hist(tfidf_word_match_share[train_df['is_duplicate'] == 1].fillna(0), bins=20, normed=True, alpha=0.7,
                 label='Duplicate')
        plt.legend()
        plt.title('Label distribution over tfidf_word_match_share', fontsize=15)
        plt.xlabel('word_match_share', fontsize=15)
        plt.show()

    @staticmethod
    def demo():
        """
        使用样例
        :return: NONE
        """
        # 读取配置文件
        cf = ConfigParser.ConfigParser()
        cf.read("../conf/python.conf")

        # 加载train.csv文件
        train_data = pd.read_csv('%s/train.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")
        # 加载test.csv文件
        test_data = pd.read_csv('%s/test.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")
        # 特征文件路径
        feature_path = cf.get('DEFAULT', 'feature_question_pair_pt')
        # 提取特征
        TFIDFWordMatchShare.run(train_data, test_data, feature_path)


class MyWordMatchShare(object):
    """
    提取<Q1,Q2>特征：my_word_match_share
    """

    stops = set(stopwords.words("english"))

    @staticmethod
    def word_match_share(row):
        """
        针对单个<Q1,Q2>实例抽取特征
        :param row: DataFrame中的一行数据
        :return: 特征值
        """
        q1words = {}
        q2words = {}
        for word in str(row['question1']).lower().split():
            if word not in WordMatchShare.stops:
                q1words[word] = q1words.get(word, 0) + 1
        for word in str(row['question2']).lower().split():
            if word not in WordMatchShare.stops:
                q2words[word] = q2words.get(word, 0) + 1
        n_shared_word_in_q1 = sum([q1words[w] for w in q1words if w in q2words])
        n_shared_word_in_q2 = sum([q2words[w] for w in q2words if w in q1words])
        n_tol = sum(q1words.values()) + sum(q2words.values())
        if (1e-6 > n_tol):
            return [0.]
        else:
            return [1.0 * (n_shared_word_in_q1 + n_shared_word_in_q2) / n_tol]

    @staticmethod
    def run(train_df, test_df, feature_pt):
        """
        抽取train.csv test.csv特征
        :param train_df: train.csv数据
        :param test_df: test.csv数据
        :param feature_pt: 特征文件路径
        :return: None
        """
        train_features = train_df.apply(MyWordMatchShare.word_match_share, axis=1, raw=True)
        Feature.save_dataframe(train_features, feature_pt + '/my_word_match_share.train.smat')

        test_features = test_df.apply(MyWordMatchShare.word_match_share, axis=1, raw=True)
        Feature.save_dataframe(test_features, feature_pt + '/my_word_match_share.test.smat')

        my_word_match_share = train_features.apply(lambda x: x[0])
        plt.figure(figsize=(15, 5))
        plt.hist(my_word_match_share[train_df['is_duplicate'] == 0], bins=20, normed=True, label='Not Duplicate')
        plt.hist(my_word_match_share[train_df['is_duplicate'] == 1], bins=20, normed=True, alpha=0.7, label='Duplicate')
        plt.legend()
        plt.title('Label distribution over word_match_share', fontsize=15)
        plt.xlabel('word_match_share', fontsize=15)
        plt.show()

    @staticmethod
    def demo():
        """
        使用样例
        :return: NONE
        """
        # 读取配置文件
        cf = ConfigParser.ConfigParser()
        cf.read("../conf/python.conf")

        # 加载train.csv文件
        train_data = pd.read_csv('%s/train.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")
        # 加载test.csv文件
        test_data = pd.read_csv('%s/test.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")
        # 特征文件路径
        feature_path = cf.get('DEFAULT', 'feature_question_pair_pt')
        # 提取特征
        MyWordMatchShare.run(train_data, test_data, feature_path)


class MyTFIDFWordMatchShare(object):
    """
    基于TF-IDF的词共享特征
    """

    idf = {}

    @staticmethod
    def init_idf(data):
        """
        根据文档计算IDF，包括停用词
        :param data: DataFrame数据
        :return: IDF词典
        """
        idf = {}
        for index, row in data.iterrows():
            words = set(str(row['question']).lower().split())
            for word in words:
                idf[word] = idf.get(word, 0) + 1
        num_docs = len(data)
        for word in idf:
            idf[word] = math.log(num_docs / (idf[word] + 1.)) / math.log(2.)
        LogUtil.log("INFO", "IDF calculation done, len(idf)=%d" % len(idf))
        MyTFIDFWordMatchShare.idf = idf

    @staticmethod
    def tfidf_word_match_share(row):
        """
        针对一个<Q1,Q2>抽取特征
        :param row: 一个<Q1,Q2>实例
        :return: 特征值
        """
        q1words = {}
        q2words = {}
        for word in str(row['question1']).lower().split():
            q1words[word] = q1words.get(word, 0) + 1
        for word in str(row['question2']).lower().split():
            q2words[word] = q2words.get(word, 0) + 1
        sum_shared_word_in_q1 = sum([q1words[w] * MyTFIDFWordMatchShare.idf.get(w, 0) for w in q1words if w in q2words])
        sum_shared_word_in_q2 = sum([q2words[w] * MyTFIDFWordMatchShare.idf.get(w, 0) for w in q2words if w in q1words])
        sum_tol = sum(q1words[w] * MyTFIDFWordMatchShare.idf.get(w, 0) for w in q1words) + sum(
            q2words[w] * MyTFIDFWordMatchShare.idf.get(w, 0) for w in q2words)
        if 1e-6 > sum_tol:
            return [0.]
        else:
            return [1.0 * (sum_shared_word_in_q1 + sum_shared_word_in_q2) / sum_tol]

    @staticmethod
    def run(train_df, test_df, train_qid2question, feature_pt):
        """
        抽取train.csv和test.csv的<Q1,Q2>特征：my_tfidf_word_match_share
        :param train_df: train.csv
        :param test_df: test.csv
        :param train_qid2question: train.csv去重question集合
        :param feature_pt: 特征文件目录
        :return: None
        """
        # 获取weights信息
        MyTFIDFWordMatchShare.init_idf(train_qid2question)
        # 抽取特征
        train_features = train_df.apply(MyTFIDFWordMatchShare.tfidf_word_match_share, axis=1, raw=True)
        Feature.save_dataframe(train_features, feature_pt + '/my_tfidf_word_match_share.train.smat')
        test_features = test_df.apply(MyTFIDFWordMatchShare.tfidf_word_match_share, axis=1, raw=True)
        Feature.save_dataframe(test_features, feature_pt + '/my_tfidf_word_match_share.test.smat')
        # 绘图
        my_tfidf_word_match_share = train_features.apply(lambda x: x[0])
        plt.figure(figsize=(15, 5))
        plt.hist(my_tfidf_word_match_share[train_df['is_duplicate'] == 0].fillna(0), bins=20, normed=True,
                 label='Not Duplicate')
        plt.hist(my_tfidf_word_match_share[train_df['is_duplicate'] == 1].fillna(0), bins=20, normed=True, alpha=0.7,
                 label='Duplicate')
        plt.legend()
        plt.title('Label distribution over tfidf_word_match_share', fontsize=15)
        plt.xlabel('word_match_share', fontsize=15)
        plt.show()

    @staticmethod
    def demo():
        """
        使用样例
        :return: NONE
        """
        # 读取配置文件
        cf = ConfigParser.ConfigParser()
        cf.read("../conf/python.conf")

        # 加载train.csv文件
        train_data = pd.read_csv('%s/train.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")
        # 加载test.csv文件
        test_data = pd.read_csv('%s/test.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")
        # 特征文件路径
        feature_path = cf.get('DEFAULT', 'feature_question_pair_pt')
        # train.csv中去重question文件
        train_qid2q_fp = '%s/train_qid2question.csv' % cf.get('DEFAULT', 'devel_pt')
        train_qid2q = pd.read_csv(train_qid2q_fp).fillna(value="")
        # 提取特征
        MyTFIDFWordMatchShare.run(train_data, test_data, train_qid2q, feature_path)


class PowerfulWord(object):
    """
    寻找最有影响力的词
    """
    dside_word_power = []
    oside_word_power = []
    aside_word_power = []
    word_power_dict = {}

    @staticmethod
    def init_word_power_dict(words_power_fp):
        """
        初始化静态成员变量 word_power_dict
        :param words_power_fp: 关键词词典路径
        :return:
        """
        words_power = PowerfulWord.load_word_power(words_power_fp)
        PowerfulWord.word_power_dict = dict(words_power)

    @staticmethod
    def init_dside_word_power(words_power):
        """
        初始化双边影响力词表
        :param words_power: 影响力词表
        :return: NONE
        """
        PowerfulWord.dside_word_power = []
        # 在双边pair中最少出现的次数
        num_least = 500
        words_power = filter(lambda x: x[1][0] * x[1][5] >= num_least, words_power)
        # 按照双侧语句对正确比例排序
        sorted_words_power = sorted(words_power, key=lambda d: d[1][6], reverse=True)
        # 双侧正确比例阈值
        dside_corate_rate = 0.7
        PowerfulWord.dside_word_power.extend(
            map(lambda x: x[0], filter(lambda x: x[1][6] >= dside_corate_rate, sorted_words_power)))
        LogUtil.log('INFO', 'Double side power words(%d): %s' % (
            len(PowerfulWord.dside_word_power), str(PowerfulWord.dside_word_power)))

    @staticmethod
    def init_oside_word_power(words_power):
        """
        初始化单边影响力词表
        :param words_power:
        :return:
        """
        PowerfulWord.oside_word_power = []
        # 在单边pair中最少出现次数
        num_least = 500
        words_power = filter(lambda x: x[1][0] * x[1][3] >= num_least, words_power)
        # 单边正确比例阈值
        oside_corate_rate = 0.9
        PowerfulWord.oside_word_power.extend(
            map(lambda x: x[0], filter(lambda x: x[1][4] >= oside_corate_rate, words_power)))
        LogUtil.log('INFO', 'One side power words(%d): %s' % (
            len(PowerfulWord.oside_word_power), str(PowerfulWord.oside_word_power)))

    @staticmethod
    def init_aside_word_power(words_power):
        PowerfulWord.aside_word_power = []
        # 在pair中最少出现次数
        num_least = 500
        words_power = filter(lambda x: x[1][0] >= num_least, words_power)
        # 按照正确语句比例排序
        sorted_words_power = sorted(words_power, key=lambda d: d[1][2], reverse=True)
        # 正确语句比例阈值
        aside_corate_rate = 0.7
        PowerfulWord.dside_word_power.extend(
            map(lambda x: x[0], filter(lambda x: x[1][2] >= aside_corate_rate, sorted_words_power)))
        LogUtil.log('INFO', 'Double side power words: %s' % str(PowerfulWord.dside_word_power))

    @staticmethod
    def cal_word_power(train_data, train_subset_indexs):
        """
        计算数据中词语的影响力，格式如下：
            词语 --> [0. 出现语句对数量，1. 出现语句对比例，2. 正确语句对比例，3. 单侧语句对比例，4. 单侧语句对正确比例，5. 双侧语句对比例，6. 双侧语句对正确比例]
        :param data: 训练数据
        :return: 影响力词典
        """
        words_power = {}
        train_subset_data = train_data.iloc[train_subset_indexs, :]
        for index, row in train_subset_data.iterrows():
            label = int(row['is_duplicate'])
            q1_words = str(row['question1']).lower().split()
            q2_words = str(row['question2']).lower().split()
            all_words = set(q1_words + q2_words)
            q1_words = set(q1_words)
            q2_words = set(q2_words)
            for word in all_words:
                if word not in words_power:
                    words_power[word] = [0. for i in range(7)]
                # 计算出现语句对数量
                words_power[word][0] += 1.
                words_power[word][1] += 1.

                if ((word in q1_words) and (word not in q2_words)) or ((word not in q1_words) and (word in q2_words)):
                    # 计算单侧语句数量
                    words_power[word][3] += 1.
                    if 0 == label:
                        # 计算正确语句对数量
                        words_power[word][2] += 1.
                        # 计算单侧语句正确比例
                        words_power[word][4] += 1.
                if (word in q1_words) and (word in q2_words):
                    # 计算双侧语句数量
                    words_power[word][5] += 1.
                    if 1 == label:
                        # 计算正确语句对数量
                        words_power[word][2] += 1.
                        # 计算双侧语句正确比例
                        words_power[word][6] += 1.
        for word in words_power:
            # 计算出现语句对比例
            words_power[word][1] /= len(train_subset_indexs)
            # 计算正确语句对比例
            words_power[word][2] /= words_power[word][0]
            # 计算单侧语句对正确比例
            if words_power[word][3] > 1e-6:
                words_power[word][4] /= words_power[word][3]
            # 计算单侧语句对比例
            words_power[word][3] /= words_power[word][0]
            # 计算双侧语句对正确比例
            if words_power[word][5] > 1e-6:
                words_power[word][6] /= words_power[word][5]
            # 计算双侧语句对比例
            words_power[word][5] /= words_power[word][0]
        sorted_words_power = sorted(words_power.iteritems(), key=lambda d: d[1][0], reverse=True)
        LogUtil.log("INFO", "power words calculation done, len(words_power)=%d" % len(sorted_words_power))
        return sorted_words_power

    @staticmethod
    def save_word_power(words_power, fp):
        """
        存储影响力词表
        :param words_power: 影响力词表
        :param fp: 存储路径
        :return: NONE
        """
        f = open(fp, 'w')
        for ele in words_power:
            f.write("%s" % ele[0])
            for num in ele[1]:
                f.write("\t%.5f" % num)
            f.write("\n")
        f.close()

    @staticmethod
    def load_word_power(fp):
        """
        加载影响力词表
        :param fp: 影响力词表路径
        :return: 影响力词表
        """
        words_power = []
        f = open(fp, 'r')
        for line in f:
            subs = line.split('\t')
            word = subs[0]
            stats = [float(num) for num in subs[1:]]
            words_power.append((word, stats))
        f.close()
        return words_power

    @staticmethod
    def tag_dside_word_power(row):
        """
        针对一个Pair抽取特征：是否包含双边影响力词表
        :param row: 一个Pair实例
        :return: Tags
        """
        tags = []
        q1_words = str(row['question1']).lower().split()
        q2_words = str(row['question2']).lower().split()
        for word in PowerfulWord.dside_word_power:
            if (word in q1_words) and (word in q2_words):
                tags.append(1.0)
            else:
                tags.append(0.0)
        return tags

    @staticmethod
    def tag_oside_word_power(row):
        """
        针对一个Pair抽取特征：是否包含单边影响力词表
        :param row:
        :return:
        """
        tags = []
        q1_words = set(str(row['question1']).lower().split())
        q2_words = set(str(row['question2']).lower().split())
        for word in PowerfulWord.oside_word_power:
            if (word in q1_words) and (word not in q2_words):
                tags.append(1.0)
            elif (word not in q1_words) and (word in q2_words):
                tags.append(1.0)
            else:
                tags.append(0.0)
        return tags

    @staticmethod
    def run_dside_word_power(train_data, test_data, words_power_fp, feature_pt):

        # 加载影响力词表
        words_power = PowerfulWord.load_word_power(words_power_fp)
        # 筛选双边影响力词表
        PowerfulWord.init_dside_word_power(words_power)

        # 抽取双边影响力词表特征
        train_features = train_data.apply(PowerfulWord.tag_dside_word_power, axis=1, raw=True)
        Feature.save_dataframe(train_features, feature_pt + '/dside_word_power.train.smat')
        test_features = test_data.apply(PowerfulWord.tag_dside_word_power, axis=1, raw=True)
        Feature.save_dataframe(test_features, feature_pt + '/dside_word_power.test.smat')

        # 统计
        neg_train_features = train_features[train_data['is_duplicate'] == 0]
        num_neg_dside_pair = 0
        for row in neg_train_features:
            if sum(row):
                num_neg_dside_pair += 1
        pos_train_features = train_features[train_data['is_duplicate'] == 1]
        num_pos_dside_pair = 0
        for row in pos_train_features:
            if sum(row):
                num_pos_dside_pair += 1
        LogUtil.log("INFO", 'train neg: sum=%.2f, train pos: sum=%.2f' % (num_neg_dside_pair, num_pos_dside_pair))
        return

    @staticmethod
    def run_oside_word_power(train_data, test_data, words_power_fp, feature_pt):

        # 加载影响力词表
        words_power = PowerfulWord.load_word_power(words_power_fp)
        # 筛选单边影响力词表
        PowerfulWord.init_oside_word_power(words_power)

        # 抽取单边影响力词表特征
        train_features = train_data.apply(PowerfulWord.tag_oside_word_power, axis=1, raw=True)
        Feature.save_dataframe(train_features, feature_pt + '/oside_word_power.train.smat')
        test_features = test_data.apply(PowerfulWord.tag_oside_word_power, axis=1, raw=True)
        Feature.save_dataframe(test_features, feature_pt + '/oside_word_power.test.smat')

        # 统计
        neg_train_features = train_features[train_data['is_duplicate'] == 0]
        num_neg_oside_pair = 0
        for row in neg_train_features:
            if sum(row):
                num_neg_oside_pair += 1
        pos_train_features = train_features[train_data['is_duplicate'] == 1]
        num_pos_oside_pair = 0
        for row in pos_train_features:
            if sum(row):
                num_pos_oside_pair += 1
        LogUtil.log("INFO", 'train neg: sum=%.2f, train pos: sum=%.2f' % (num_neg_oside_pair, num_pos_oside_pair))
        return

    @staticmethod
    def tag_any_dside_word_power(row):
        """
        针对一个Pair抽取特征：是否包含任一双边影响力词表
        :param row: 一个Pair实例
        :return: Tag
        """
        tag = [0.0]
        q1_words = str(row['question1']).lower().split()
        q2_words = str(row['question2']).lower().split()
        for word in PowerfulWord.dside_word_power:
            if (word in q1_words) and (word in q2_words):
                tag[0] = 1.0
                break
        return tag

    @staticmethod
    def run_any_dside_word_power(train_data, test_data, words_power_fp, feature_pt):
        """
        计算train.csv test.csv特征：any_dside_word_power
        :param train_data: train.csv数据
        :param test_data: test.csv数据
        :param words_power_fp:
        :param feature_pt:
        :return:
        """
        # 加载影响力词表
        words_power = PowerfulWord.load_word_power(words_power_fp)
        # 筛选双边影响力词表
        PowerfulWord.init_dside_word_power(words_power)

        # 抽取双边影响力词表特征
        train_features = train_data.apply(PowerfulWord.tag_any_dside_word_power, axis=1, raw=True)
        Feature.save_dataframe(train_features, feature_pt + '/any_dside_word_power.train.smat')
        test_features = test_data.apply(PowerfulWord.tag_any_dside_word_power, axis=1, raw=True)
        Feature.save_dataframe(test_features, feature_pt + '/any_dside_word_power.test.smat')

        # 统计
        neg_train_features = train_features[train_data['is_duplicate'] == 0]
        num_neg_dside_pair = 0
        for row in neg_train_features:
            if sum(row):
                num_neg_dside_pair += 1
        pos_train_features = train_features[train_data['is_duplicate'] == 1]
        num_pos_dside_pair = 0
        for row in pos_train_features:
            if sum(row):
                num_pos_dside_pair += 1
        LogUtil.log("INFO", 'train neg: sum=%.2f, train pos: sum=%.2f' % (num_neg_dside_pair, num_pos_dside_pair))
        return

    @staticmethod
    def cal_rate_by_dside_word_power(row):
        """
        针对一个Pair抽取特征：rate_by_dside_word_power
        :param row:
        :return:
        """
        num_least = 300
        rate = [1.0]
        q1_words = set(str(row['question1']).lower().split())
        q2_words = set(str(row['question2']).lower().split())
        share_words = list(q1_words.intersection(q2_words))
        for word in share_words:
            if word not in PowerfulWord.word_power_dict:
                continue
            if PowerfulWord.word_power_dict[word][0] * PowerfulWord.word_power_dict[word][5] < num_least:
                continue
            rate[0] *= (1.0 - PowerfulWord.word_power_dict[word][6])
        rate = [1 - num for num in rate]
        return rate

    @staticmethod
    def run_rate_by_dside_word_power(train_data, test_data, words_power_fp, feature_pt):
        # 初始化
        PowerfulWord.init_word_power_dict(words_power_fp)

        # 抽取特征
        train_features = train_data.apply(PowerfulWord.cal_rate_by_dside_word_power, axis=1, raw=True)
        Feature.save_dataframe(train_features, feature_pt + '/rate_by_dside_word_power.train.smat')
        test_features = test_data.apply(PowerfulWord.cal_rate_by_dside_word_power, axis=1, raw=True)
        Feature.save_dataframe(test_features, feature_pt + '/rate_by_dside_word_power.test.smat')

        # 统计
        # 负例
        neg_train_features = train_features[train_data['is_duplicate'] == 0].apply(lambda x: x[0])
        LogUtil.log("INFO", 'neg: mean=%.2f, std=%.2f, max=%.2f, min=%.2f' % (
            neg_train_features.mean(), neg_train_features.std(), neg_train_features.max(), neg_train_features.min()))
        # 正例
        pos_train_features = train_features[train_data['is_duplicate'] == 1].apply(lambda x: x[0])
        LogUtil.log("INFO", 'pos: mean=%.2f, std=%.2f, max=%.2f, min=%.2f' % (
            pos_train_features.mean(), pos_train_features.std(), pos_train_features.max(), pos_train_features.min()))

        # 绘图
        plt.figure(figsize=(15, 5))
        plt.xlim([0, 1])
        plt.hist(neg_train_features, bins=50, normed=False, label='Not Duplicate', edgecolor='None')
        plt.hist(pos_train_features, bins=50, normed=False, alpha=0.7, label='Duplicate', edgecolor='None')
        plt.legend()
        plt.title('Label distribution over rate_by_dside_word_power', fontsize=15)
        plt.xlabel('rate_by_dside_word_power', fontsize=15)
        plt.show()

    @staticmethod
    def cal_rate_by_oside_word_power(row):
        """
        针对一个Pair抽取特征：rate_by_oside_word_power
        :param row:
        :return:
        """
        num_least = 300
        rate = [1.0]
        q1_words = set(str(row['question1']).lower().split())
        q2_words = set(str(row['question2']).lower().split())
        q1_diff = list(set(q1_words).difference(set(q2_words)))
        q2_diff = list(set(q2_words).difference(set(q1_words)))
        all_diff = set(q1_diff + q2_diff)
        for word in all_diff:
            if word not in PowerfulWord.word_power_dict:
                continue
            if PowerfulWord.word_power_dict[word][0] * PowerfulWord.word_power_dict[word][3] < num_least:
                continue
            rate[0] *= (1.0 - PowerfulWord.word_power_dict[word][4])
        rate = [1 - num for num in rate]
        return rate

    @staticmethod
    def generate_word_power(train_data, train_subset_indexs, words_power_fp):
        """
        生成并存储影响力词表
        :param train_data:
        :param train_subset_indexs:
        :param words_power_fp:
        :return:
        """
        # 计算词语影响力
        words_power = PowerfulWord.cal_word_power(train_data, train_subset_indexs)
        # 存储影响力词表
        PowerfulWord.save_word_power(words_power, words_power_fp)

    @staticmethod
    def run_rate_by_oside_word_power(train_data, test_data, words_power_fp, feature_pt):
        # 初始化
        PowerfulWord.init_word_power_dict(words_power_fp)

        # 抽取特征
        train_features = train_data.apply(PowerfulWord.cal_rate_by_oside_word_power, axis=1, raw=True)
        Feature.save_dataframe(train_features, feature_pt + '/rate_by_oside_word_power.train.smat')
        test_features = test_data.apply(PowerfulWord.cal_rate_by_oside_word_power, axis=1, raw=True)
        Feature.save_dataframe(test_features, feature_pt + '/rate_by_oside_word_power.test.smat')

        # 统计
        # 负例
        neg_train_features = train_features[train_data['is_duplicate'] == 0].apply(lambda x: x[0])
        LogUtil.log("INFO", 'neg: mean=%.2f, std=%.2f, max=%.2f, min=%.2f' % (
            neg_train_features.mean(), neg_train_features.std(), neg_train_features.max(), neg_train_features.min()))
        # 正例
        pos_train_features = train_features[train_data['is_duplicate'] == 1].apply(lambda x: x[0])
        LogUtil.log("INFO", 'pos: mean=%.2f, std=%.2f, max=%.2f, min=%.2f' % (
            pos_train_features.mean(), pos_train_features.std(), pos_train_features.max(), pos_train_features.min()))

        # 绘图
        plt.figure(figsize=(15, 5))
        plt.xlim([0, 1])
        plt.hist(neg_train_features, bins=100, normed=False, label='Not Duplicate', edgecolor='None')
        plt.hist(pos_train_features, bins=100, normed=False, alpha=0.7, label='Duplicate', edgecolor='None')
        plt.legend()
        plt.title('Label distribution over rate_by_oside_word_power', fontsize=15)
        plt.xlabel('rate_by_oside_word_power', fontsize=15)
        plt.show()

    @staticmethod
    def run_aside_word_power(train_data, test_data, words_power_fp, feature_pt):
        # 加载影响力词表
        words_power = PowerfulWord.load_word_power(words_power_fp)
        # 筛选双边影响力词表
        PowerfulWord.init_aside_word_power(words_power)

    @staticmethod
    def run():
        # 读取配置文件
        cf = ConfigParser.ConfigParser()
        cf.read("../conf/python.conf")

        # 加载stem.train.csv文件
        train_stem_data = pd.read_csv('%s/stem.train.csv' % cf.get('DEFAULT', 'devel_pt')).fillna(value="")
        # 加载训练子集索引文件（NOTE: 不是训练集）
        train_subset_indexs = Feature.load_index(cf.get('MODEL', 'train_indexs_fp'))
        # 影响力词表路径
        words_power_stem_fp = '%s/words_power.stem.%s.%s.txt' % (
            cf.get('DEFAULT', 'feature_stat_pt'),
            cf.get('MODEL', 'train_subset_name'),
            cf.get('MODEL', 'train_rawset_name'))

        # 生成影响力词表
        PowerfulWord.generate_word_power(train_stem_data, train_subset_indexs, words_power_stem_fp)

    @staticmethod
    def demo():
        """
        使用示例
        :return:
        """
        # 读取配置文件
        cf = ConfigParser.ConfigParser()
        cf.read("../conf/python.conf")

        # 加载train.csv文件
        train_data = pd.read_csv('%s/train.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")
        # 加载训练子集索引文件（NOTE: 不是训练集）
        train_subset_indexs = Feature.load_index(cf.get('MODEL', 'train_indexs_fp'))
        # 影响力词表路径
        words_power_fp = '%s/words_power.%s.txt' % (
            cf.get('DEFAULT', 'feature_stat_pt'), cf.get('MODEL', 'train_subset_name'))
        # 加载test.csv文件
        test_data = pd.read_csv('%s/test.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")
        # 特征文件路径
        feature_path = cf.get('DEFAULT', 'feature_question_pair_pt')

        # 提取特征
        PowerfulWord.run_oside_word_power(train_data, test_data, words_power_fp, feature_path)


class QuestionLenDiff(object):
    """
    <Q1, Q2>长度差特征：
        1.  长度差
        2.  len(短句) / len(长句)
    """

    @staticmethod
    def cal_len_diff(row):
        """
        针对一个Pair抽取特征：长度差的绝对值
        :param row: 一个Pair实例
        :return: 特征值
        """
        q1 = row['question1']
        q2 = row['question2']
        return [abs(len(q1) - len(q2))]

    @staticmethod
    def run_len_diff(train_df, test_df, feature_pt):
        """
        抽取train.csv和test.csv的Pair特征：ABS(长度差)
        :param train_df: train.csv
        :param test_df: test.csv
        :param feature_pt: 特征文件目录
        :return: NONE
        """
        # 抽取特征ABS(长度差)
        train_features = train_df.apply(QuestionLenDiff.cal_len_diff, axis=1, raw=True)
        Feature.save_dataframe(train_features, feature_pt + '/len_diff.train.smat')
        test_features = test_df.apply(QuestionLenDiff.cal_len_diff, axis=1, raw=True)
        Feature.save_dataframe(test_features, feature_pt + '/len_diff.test.smat')

        # 统计
        # 负例
        neg_train_features = train_features[train_df['is_duplicate'] == 0].apply(lambda x: x[0])
        LogUtil.log("INFO", 'neg: mean=%.2f, std=%.2f, max=%.2f, min=%.2f' % (
            neg_train_features.mean(), neg_train_features.std(), neg_train_features.max(), neg_train_features.min()))
        # 正例
        pos_train_features = train_features[train_df['is_duplicate'] == 1].apply(lambda x: x[0])
        LogUtil.log("INFO", 'pos: mean=%.2f, std=%.2f, max=%.2f, min=%.2f' % (
            pos_train_features.mean(), pos_train_features.std(), pos_train_features.max(), pos_train_features.min()))

        # 绘图
        plt.figure(figsize=(15, 5))
        plt.xlim([0, 200])
        plt.hist(neg_train_features, bins=200, normed=False, label='Not Duplicate', edgecolor='None')
        plt.hist(pos_train_features, bins=200, normed=False, alpha=0.7, label='Duplicate', edgecolor='None')
        plt.legend()
        plt.title('Label distribution over len_diff', fontsize=15)
        plt.xlabel('len_diff', fontsize=15)
        plt.show()

    @staticmethod
    def cal_len_diff_rate(row):
        """
        针对一个Pair抽取特征：len_short / len_long
        :param row: 一个Pair实例
        :return: 特征值
        """
        len_q1 = len(row['question1'])
        len_q2 = len(row['question2'])
        if max(len_q1, len_q2) < 1e-6:
            return [0.0]
        else:
            return [1.0 * min(len_q1, len_q2) / max(len_q1, len_q2)]

    @staticmethod
    def run_len_diff_rate(train_df, test_df, feature_pt):
        """
        抽取train.csv和test.csv的Pair特征：len_short / len_long
        :param train_df: train.csv
        :param test_df: test.csv
        :param feature_pt: 特征文件目录
        :return: NONE
        """
        # 抽取特征
        train_features = train_df.apply(QuestionLenDiff.cal_len_diff_rate, axis=1, raw=True)
        Feature.save_dataframe(train_features, feature_pt + '/len_diff_rate.train.smat')
        test_features = test_df.apply(QuestionLenDiff.cal_len_diff_rate, axis=1, raw=True)
        Feature.save_dataframe(test_features, feature_pt + '/len_diff_rate.test.smat')

        # 统计
        # 负例
        neg_train_features = train_features[train_df['is_duplicate'] == 0].apply(lambda x: x[0])
        LogUtil.log("INFO", 'neg: mean=%.2f, std=%.2f, max=%.2f, min=%.2f' % (
            neg_train_features.mean(), neg_train_features.std(), neg_train_features.max(), neg_train_features.min()))
        # 正例
        pos_train_features = train_features[train_df['is_duplicate'] == 1].apply(lambda x: x[0])
        LogUtil.log("INFO", 'pos: mean=%.2f, std=%.2f, max=%.2f, min=%.2f' % (
            pos_train_features.mean(), pos_train_features.std(), pos_train_features.max(), pos_train_features.min()))

        # 绘图
        plt.figure(figsize=(15, 5))
        plt.xlim([0, 1])
        plt.hist(neg_train_features, bins=200, normed=False, label='Not Duplicate', edgecolor='None')
        plt.hist(pos_train_features, bins=200, normed=False, alpha=0.7, label='Duplicate', edgecolor='None')
        plt.legend()
        plt.title('Label distribution over len_diff_rate', fontsize=15)
        plt.xlabel('len_diff_rate', fontsize=15)
        plt.show()

    @staticmethod
    def run(train_df, test_df, feature_pt):
        """
        抽取train.csv和test.csv的Pair特征：
            1.  ABS(长度差)
            2.  len(short_question) / len(long_question)
        :param train_df: train.csv
        :param test_df: test.csv
        :param feature_pt: 特征文件目录
        :return: NONE
        """
        # 抽取特征 ABS(长度差)
        # QuestionLenDiff.run_len_diff(train_df, test_df, feature_pt)
        # 抽取特征 len_short / len_long
        QuestionLenDiff.run_len_diff_rate(train_df, test_df, feature_pt)

    @staticmethod
    def demo():
        """
        使用样例
        :return: NONE
        """
        # 读取配置文件
        cf = ConfigParser.ConfigParser()
        cf.read("../conf/python.conf")

        # 加载train.csv文件
        train_data = pd.read_csv('%s/train.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")
        # 加载test.csv文件
        test_data = pd.read_csv('%s/test.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")
        # 特征文件路径
        feature_path = cf.get('DEFAULT', 'feature_question_pair_pt')
        # 提取特征
        QuestionLenDiff.run(train_data, test_data, feature_path)


class F00FromKaggle(object):
    """
    Kaggle解决方案：https://www.kaggle.com/the1owl/quora-question-pairs/matching-que-for-quora-end-to-end-0-33719-pb
    """

    tfidf = None
    seq = difflib.SequenceMatcher()

    @staticmethod
    def init_tfidf(train_data, test_data):
        F00FromKaggle.tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 1))
        tfidf_txt = pd.Series(
            train_data['question1'].tolist() + train_data['question2'].tolist() + test_data['question1'].tolist() +
            test_data['question2'].tolist()).astype(str)
        F00FromKaggle.tfidf.fit_transform(tfidf_txt)
        LogUtil.log("INFO", "init tfidf done ")

    @staticmethod
    def diff_ratios(st1, st2):
        seq = difflib.SequenceMatcher()
        seq.set_seqs(str(st1).lower(), str(st2).lower())
        return seq.ratio()

    @staticmethod
    def extract(data):
        # about nouns
        data['question1_nouns'] = data.question1.map(
            lambda x: [w for w, t in nltk.pos_tag(nltk.word_tokenize(str(x).lower().decode('utf-8'))) if
                       t[:1] in ['N']])
        LogUtil.log('INFO', 'question1_nouns done')
        data['question2_nouns'] = data.question2.map(
            lambda x: [w for w, t in nltk.pos_tag(nltk.word_tokenize(str(x).lower().decode('utf-8'))) if
                       t[:1] in ['N']])
        LogUtil.log('INFO', 'question2_nouns done')
        data['z_noun_match'] = data.apply(
            lambda r: sum([1 for w in r.question1_nouns if w in r.question2_nouns]), axis=1)  # takes long
        LogUtil.log('INFO', 'z_noun_match done')
        # about length
        data['z_len1'] = data.question1.map(lambda x: len(str(x)))
        LogUtil.log('INFO', 'z_len1 done')
        data['z_len2'] = data.question2.map(lambda x: len(str(x)))
        LogUtil.log('INFO', 'z_len2 done')
        data['z_word_len1'] = data.question1.map(lambda x: len(str(x).split()))
        LogUtil.log('INFO', 'z_word_len1 done')
        data['z_word_len2'] = data.question2.map(lambda x: len(str(x).split()))
        LogUtil.log('INFO', 'z_word_len2 done')
        # about difflib
        data['z_match_ratio'] = data.apply(lambda r: F00FromKaggle.diff_ratios(r.question1, r.question2),
                                           axis=1)  # takes long
        LogUtil.log('INFO', 'z_noun_match done')
        # abount tfidf
        data['z_tfidf_sum1'] = data.question1.map(lambda x: np.sum(F00FromKaggle.tfidf.transform([str(x)]).data))
        LogUtil.log('INFO', 'z_tfidf_sum1 done')
        data['z_tfidf_sum2'] = data.question2.map(lambda x: np.sum(F00FromKaggle.tfidf.transform([str(x)]).data))
        LogUtil.log('INFO', 'z_tfidf_sum2 done')
        data['z_tfidf_mean1'] = data.question1.map(lambda x: np.mean(F00FromKaggle.tfidf.transform([str(x)]).data))
        LogUtil.log('INFO', 'z_tfidf_mean1 done')
        data['z_tfidf_mean2'] = data.question2.map(lambda x: np.mean(F00FromKaggle.tfidf.transform([str(x)]).data))
        LogUtil.log('INFO', 'z_tfidf_mean2 done')
        data['z_tfidf_len1'] = data.question1.map(lambda x: len(F00FromKaggle.tfidf.transform([str(x)]).data))
        LogUtil.log('INFO', 'z_tfidf_len1 done')
        data['z_tfidf_len2'] = data.question2.map(lambda x: len(F00FromKaggle.tfidf.transform([str(x)]).data))
        LogUtil.log('INFO', 'z_tfidf_len2 done')

        # fulfill nan
        data = data.fillna(0.0)

        # get features
        features = data.apply(lambda x: [float(x[cn]) for cn in data.columns if cn[:1] == 'z'], axis=1)
        return features

    @staticmethod
    def run(train_data, test_data, feature_pt):
        F00FromKaggle.init_tfidf(train_data, test_data)

        train_features = F00FromKaggle.extract(train_data)
        Feature.save_dataframe(train_features, feature_pt + '/f00_from_kaggle.train.smat')
        test_features = F00FromKaggle.extract(test_data)
        Feature.save_dataframe(test_features, feature_pt + '/f00_from_kaggle.test.smat')

    @staticmethod
    def demo():
        """
        使用示例
        :return:
        """
        # 读取配置文件
        cf = ConfigParser.ConfigParser()
        cf.read("../conf/python.conf")

        # 加载train.csv文件
        train_data = pd.read_csv('%s/train.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")  # [:100]
        # 加载test.csv文件
        test_data = pd.read_csv('%s/test.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")  # [:100]
        # 特征文件路径
        feature_path = cf.get('DEFAULT', 'feature_question_pair_pt')

        # 提取特征
        F00FromKaggle.run(train_data, test_data, feature_path)


class TreeParser(object):
    """
    使用Stanford Parser对语句进行解析
    特征：
        1.  叶子节点个数
        2.  树深度
        3.  根节点分叉数
        4.  最大分叉数
    """

    questions_features = None

    @staticmethod
    def extract_questions_ind_multi(tree_fp):
        features = {}
        f = open(tree_fp)
        for line in f:
            [qid, json_s] = line.split(' ', 1)
            features[qid] = []
            parent = {}
            indegree = {}
            # 计算入度和父节点
            if 0 < len(json_s.strip()):
                tree_obj = json.loads(json_s)
                assert len(tree_obj) <= 1
                tree_obj = tree_obj[0]
                for k, r in sorted(tree_obj.items(), key=lambda x: int(x[0]))[1:]:
                    if r['word'] is None:
                        continue
                    head = int(r['head'])
                    k = int(k)
                    if 0 == head:
                        root = k
                    parent[k] = head
                    indegree[head] = indegree.get(head, 0) + 1
            # 计算入度乘积
            ind_multi = 1.0
            for id_node in indegree:
                ind_multi *= indegree[id_node]
            features[str(qid)] = [ind_multi]
        f.close()
        return features

    @staticmethod
    def extract_row_ind_multi(row):
        q1_id = str(row['qid1'])
        q2_id = str(row['qid2'])
        q1_features = TreeParser.questions_features[q1_id]
        q2_features = TreeParser.questions_features[q2_id]
        sum_features = (np.array(q1_features) + np.array(q2_features)).tolist()
        sub_features = abs(np.array(q1_features) - np.array(q2_features)).tolist()
        div_features = (np.array(q1_features) / (np.array(q2_features) + 1.)).tolist()
        mul_features = (np.array(q1_features) * (np.array(q2_features) + 0.)).tolist()
        features = q1_features + q2_features + sum_features + sub_features + div_features + mul_features
        return features

    @staticmethod
    def extract_ind_multi(data, tree_fp):
        TreeParser.questions_features = TreeParser.extract_questions_ind_multi(tree_fp)
        LogUtil.log('INFO', 'extract questions features done (%s)' % tree_fp)
        features = data.apply(TreeParser.extract_row_ind_multi, axis=1, raw=True)
        LogUtil.log('INFO', 'extract data features done, len(features)=%d' % len(features))
        return features

    @staticmethod
    def run_ind_multi(train_df, test_df, feature_pt, train_tree_fp, test_tree_fp):
        """
        抽取特征，语法树入度乘积，及加减乘除变化
        :param train_df:
        :param test_df:
        :param feature_pt:
        :param train_tree_fp:
        :param test_tree_fp:
        :return:
        """
        train_features = TreeParser.extract_ind_multi(train_df, train_tree_fp)
        Feature.save_dataframe(train_features, feature_pt + '/ind_multi.train.smat')

        test_features = TreeParser.extract_ind_multi(test_df, test_tree_fp)
        Feature.save_dataframe(test_features, feature_pt + '/ind_multi.test.smat')

    @staticmethod
    def extract_questions_features(tree_fp):
        features = {}
        f = open(tree_fp)
        for line in f:
            [qid, json_s] = line.split(' ', 1)
            # print 'qid=%s' % qid
            features[qid] = []
            root = -1
            parent = {}
            indegree = {}
            # print 'len(json_s=%d)' % len(json_s.strip())
            # 计算入度和父节点
            if 0 < len(json_s.strip()):
                tree_obj = json.loads(json_s)
                assert len(tree_obj) <= 1
                tree_obj = tree_obj[0]
                for k, r in sorted(tree_obj.items(), key=lambda x: int(x[0]))[1:]:
                    if r['word'] is None:
                        continue
                    head = int(r['head'])
                    k = int(k)
                    if 0 == head:
                        root = k
                    # print '%s --> %s' % (k, head)
                    parent[k] = head
                    indegree[head] = indegree.get(head, 0) + 1
            # 根节点
            # print 'root=%d' % root
            # print parent
            # 计算叶子节点个数
            n_child = 0
            for i in parent:
                if i not in indegree:
                    n_child += 1
            # print 'n_child=%d' % n_child
            # 计算树深度
            depth = 0
            for i in parent:
                if i not in indegree:
                    temp_id = i
                    temp_depth = 0
                    while (temp_id in parent) and (0 != parent[temp_id]):
                        temp_depth += 1
                        temp_id = parent[temp_id]
                        # print '\t%d' % temp_id,
                    depth = max(depth, temp_depth)
                    # print ''
            # print 'depth=%d' % depth
            # 计算根节点分叉数目
            n_root_braches = indegree.get(root, 0)
            # print 'n_root_braches=%d' % n_root_braches
            # 计算最大分叉数目
            n_max_braches = 0
            if 0 < len(indegree):
                n_max_braches = max(indegree.values())
            # print 'n_max_braches=%d' % n_max_braches
            features[str(qid)] = [n_child, depth, n_root_braches, n_max_braches]
        f.close()
        return features

    @staticmethod
    def extract_feature(row):
        q1_id = str(row['qid1'])
        q2_id = str(row['qid2'])
        # print q1_id
        # print q2_id
        q1_features = TreeParser.questions_features[q1_id]
        q2_features = TreeParser.questions_features[q2_id]

        return q1_features + q2_features + abs(np.array(q1_features) - np.array(q2_features)).tolist()

    @staticmethod
    def extract_features(data):
        assert TreeParser.questions_features is not None, "TreeParser.questions_features is None"
        features = data.apply(TreeParser.extract_feature, axis=1, raw=True)
        return features

    @staticmethod
    def run_tree_parser(train_df, test_df, feature_pt, train_tree_fp, test_tree_fp):
        """
        抽取特征
        :param trian_df:
        :param test_df:
        :param feature_pt:
        :return:
        """
        TreeParser.questions_features = TreeParser.extract_questions_features(train_tree_fp)
        LogUtil.log('INFO', 'extract train questions features done')
        train_features = TreeParser.extract_features(train_df)
        LogUtil.log('INFO', 'extract train features done')
        Feature.save_dataframe(train_features, feature_pt + '/tree_parser.train.smat')

        TreeParser.questions_features = TreeParser.extract_questions_features(test_tree_fp)
        LogUtil.log('INFO', 'extract test questions features done')
        test_features = TreeParser.extract_features(test_df)
        LogUtil.log('INFO', 'extract test features done')
        Feature.save_dataframe(test_features, feature_pt + '/tree_parser.test.smat')

    @staticmethod
    def demo():
        # # 读取配置文件
        # cf = ConfigParser.ConfigParser()
        # cf.read("../conf/python.conf")
        #
        # # 加载train.csv文件
        # train_data = pd.read_csv('%s/train.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")  # [:100]
        # # 加载test.csv文件
        # test_data = pd.read_csv('%s/test_with_qid.csv' % cf.get('DEFAULT', 'devel_pt')).fillna(value="")  # [:100]
        # # 特征文件路径
        # feature_path = cf.get('DEFAULT', 'feature_question_pair_pt')
        # # TreeParser文件路径
        # train_tree_fp = '%s/train_qid_query_detparse.txt' % cf.get('DEFAULT', 'devel_pt')
        # test_tree_fp = '%s/test_qid_query_detparse.txt' % cf.get('DEFAULT', 'devel_pt')
        #
        # # 提取特征
        # TreeParser.run_tree_parser(train_data, test_data, feature_path, train_tree_fp, test_tree_fp)
        # TreeParser.run_ind_multi(train_data, test_data, feature_path, train_tree_fp, test_tree_fp)

        # 读取配置文件
        cf = ConfigParser.ConfigParser()
        cf.read("../conf/python.conf")

        # 加载train.csv文件
        train_swap_data = pd.read_csv('%s/train_swap.csv' % cf.get('DEFAULT', 'devel_pt')).fillna(value="")  # [:100]
        # 特征文件路径
        feature_path = cf.get('DEFAULT', 'feature_question_pair_pt')
        # TreeParser文件路径
        train_tree_fp = '%s/train_qid_query_detparse.txt' % cf.get('DEFAULT', 'devel_pt')

        # 提取特征
        TreeParser.questions_features = TreeParser.extract_questions_ind_multi(train_tree_fp)
        LogUtil.log('INFO', 'extract questions features done (%s)' % train_tree_fp)
        features = train_swap_data.apply(TreeParser.extract_row_ind_multi, axis=1, raw=True)
        LogUtil.log('INFO', 'extract data features done, len(features)=%d' % len(features))

        Feature.save_dataframe(features, feature_path + '/ind_multi.train_swap.smat')


class F01FromKaggle(object):
    """
    Kaggle解决方案：https://www.kaggle.com/sudalairajkumar/quora-question-pairs/simple-exploration-notebook-quora-ques-pair
    """


class BTM(object):
    btm_features = {}

    @staticmethod
    def load_questions_btm(qid_fp, qf_fp):
        fqid = open(qid_fp, 'r')
        qids = fqid.readlines()
        fqid.close()

        fqf = open(qf_fp, 'r')
        qfs = fqf.readlines()
        fqf.close()

        assert len(qids) == len(qfs), "len(qid) != len(question)"

        features = {}
        for index in range(len(qids)):
            features[str(qids[index]).strip()] = qfs[index].strip()

        return features

    @staticmethod
    def extract_row_btm(row):
        q1_id = str(row['qid1'])
        q2_id = str(row['qid2'])

        q1_features = BTM.btm_features[q1_id].split()
        q2_features = BTM.btm_features[q2_id].split()

        return q1_features + q2_features

    @staticmethod
    def extract_btm(data):
        features = data.apply(BTM.extract_row_btm, axis=1, raw=True)
        LogUtil.log('INFO', 'extract btm features done, len(features)=%d' % len(features))
        return features

    @staticmethod
    def run_btm(train_data, test_data, train_feature_fp, test_feature_fp, questions_btm_qid_fp, questions_btm_qf_fp):
        BTM.btm_features = BTM.load_questions_btm(questions_btm_qid_fp, questions_btm_qf_fp)
        LogUtil.log('INFO', 'load questions btm feature done')

        train_features = BTM.extract_btm(train_data)
        LogUtil.log('INFO', 'extract btm from train data done')
        Feature.save_dataframe(train_features, train_feature_fp)

        test_features = BTM.extract_btm(test_data)
        LogUtil.log('INFO', 'extract btm from test data done')
        Feature.save_dataframe(test_features, test_feature_fp)

    @staticmethod
    def run(argv):
        feature_name = ''

        try:
            opts, args = getopt.getopt(argv[1:], 'f:', ['fname='])
        except getopt.GetoptError:
            print 'BTM.run -f <feature_name>'
            sys.exit(2)
        for opt, arg in opts:
            if opt in ('-f', '--fname'):
                feature_name = arg

        LogUtil.log('INFO', 'extractor run for BTM (%s)' % feature_name)
        # 读取配置文件
        cf = ConfigParser.ConfigParser()
        cf.read("../conf/python.conf")

        train_data = pd.read_csv('%s/train.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")  # [:100]
        train_swap_data = pd.read_csv('%s/train_swap.csv' % cf.get('DEFAULT', 'devel_pt')).fillna(value="")  # [:100]
        test_data = pd.read_csv('%s/test_with_qid.csv' % cf.get('DEFAULT', 'devel_pt')).fillna(value="")  # [:100]

        questions_btm_qid_fp = '%s/qid2question.all.qid' % cf.get('DEFAULT', 'devel_pt')
        questions_btm_qf_fp = '%s/%s.all.qf' % (cf.get('DEFAULT', 'devel_pt'), feature_name)

        feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
        train_feature_fp = '%s/%s.train.smat' % (feature_pt, feature_name)
        train_swap_feature_fp = '%s/%s.train_swap.smat' % (feature_pt, feature_name)
        test_feature_fp = '%s/%s.test.smat' % (feature_pt, feature_name)

        BTM.btm_features = BTM.load_questions_btm(questions_btm_qid_fp, questions_btm_qf_fp)
        LogUtil.log('INFO', 'load questions btm feature done (%s)' % questions_btm_qf_fp)

        train_features = BTM.extract_btm(train_data)
        LogUtil.log('INFO', 'extract btm from train data done')
        Feature.save_dataframe(train_features, train_feature_fp)

        train_swap_features = BTM.extract_btm(train_swap_data)
        LogUtil.log('INFO', 'extract btm from train_swap data done')
        Feature.save_dataframe(train_swap_features, train_swap_feature_fp)

        test_features = BTM.extract_btm(test_data)
        LogUtil.log('INFO', 'extract btm from test data done')
        Feature.save_dataframe(test_features, test_feature_fp)


class WordEmbedding(object):
    idf = {}
    we_dict = {}
    to_lower = True
    len_vec = 300

    def __init__(self):
        pass

    @staticmethod
    def init_idf(data):
        """
        根据文档计算IDF，包括停用词
        :param data:
        :return:
        """
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
        WordEmbedding.idf = idf

    @staticmethod
    def load_word_embedding(fp):
        """
        加载 Map(word, vector) 词典
        :param fp:
        :return:
        """
        we_dic = {}
        f = open(fp, 'r')
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

    @staticmethod
    def extract_row_ave_dis(row):
        """
        按行抽取特征
        :param row:
        :return:
        """
        q1_words = str(row['question1']).strip().split() if WordEmbedding.to_lower else str(
            row['question1']).lower().strip().split()
        q2_words = str(row['question2']).strip().split() if WordEmbedding.to_lower else str(
            row['question2']).lower().strip().split()

        q1_vec = np.array(WordEmbedding.len_vec * [0.])
        q2_vec = np.array(WordEmbedding.len_vec * [0.])

        for word in q1_words:
            if word in WordEmbedding.we_dict:
                q1_vec = q1_vec + WordEmbedding.we_dict[word]
        for word in q2_words:
            if word in WordEmbedding.we_dict:
                q2_vec = q2_vec + WordEmbedding.we_dict[word]

        cos_sim = 0.

        q1_vec = np.mat(q1_vec)
        q2_vec = np.mat(q2_vec)

        factor = linalg.norm(q1_vec) * linalg.norm(q2_vec)
        if 1e-6 < factor:
            cos_sim = float(q1_vec * q2_vec.T) / factor

        return [cos_sim]

    @staticmethod
    def extract_row_tfidf_dis(row):
        """
        按行抽取特征
        :param row:
        :return:
        """
        q1_words = str(row['question1']).strip().split() if 'True' == WordEmbedding.to_lower else str(
            row['question1']).lower().strip().split()
        q2_words = str(row['question2']).strip().split() if 'True' == WordEmbedding.to_lower else str(
            row['question2']).lower().strip().split()

        q1_vec = np.array(WordEmbedding.len_vec * [0.])
        q2_vec = np.array(WordEmbedding.len_vec * [0.])

        q1_words_cnt = {}
        q2_words_cnt = {}
        for word in q1_words:
            q1_words_cnt[word] = q1_words_cnt.get(word, 0.) + 1.
        for word in q2_words:
            q2_words_cnt[word] = q2_words_cnt.get(word, 0.) + 1.

        for word in q1_words_cnt:
            if word in WordEmbedding.we_dict:
                q1_vec = q1_vec + WordEmbedding.idf.get(word, 0.) * q1_words_cnt[word] * WordEmbedding.we_dict[word]
        for word in q2_words_cnt:
            if word in WordEmbedding.we_dict:
                q2_vec = q2_vec + WordEmbedding.idf.get(word, 0.) * q2_words_cnt[word] * WordEmbedding.we_dict[word]

        cos_sim = 0.

        q1_vec = np.mat(q1_vec)
        q2_vec = np.mat(q2_vec)

        factor = linalg.norm(q1_vec) * linalg.norm(q2_vec)
        if 1e-6 < factor:
            cos_sim = float(q1_vec * q2_vec.T) / factor

        return [cos_sim]

    @staticmethod
    def extract_ave_dis(cf, argv):
        # 运行需要设置的参数
        word_embedding_fp = argv[0]  # word embedding 路径
        feature_name = argv[1]  # 特征名字
        WordEmbedding.len_vec = int(argv[2])  # word embedding 维度
        WordEmbedding.to_lower = bool(argv[3])  # 是否需要转化为小写

        # 加载 word embedding 词典
        WordEmbedding.we_dict = WordEmbedding.load_word_embedding(word_embedding_fp)
        LogUtil.log('INFO', 'load word embedding dict done')

        train_data = pd.read_csv('%s/train.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")
        test_data = pd.read_csv('%s/test_with_qid.csv' % cf.get('DEFAULT', 'devel_pt')).fillna(value="")

        feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
        train_feature_fp = '%s/%s.train.smat' % (feature_pt, feature_name)
        test_feature_fp = '%s/%s.test.smat' % (feature_pt, feature_name)

        train_features = train_data.apply(WordEmbedding.extract_row_ave_dis, axis=1, raw=True)
        LogUtil.log('INFO', 'extract train features done')
        test_features = test_data.apply(WordEmbedding.extract_row_ave_dis, axis=1, raw=True)
        LogUtil.log('INFO', 'extract test features done')

        Feature.save_dataframe(train_features, train_feature_fp)
        LogUtil.log('INFO', 'save train features done')
        Feature.save_dataframe(test_features, test_feature_fp)
        LogUtil.log('INFO', 'save test features done')

    @staticmethod
    def extract_tfidf_dis(cf, argv):
        # 运行需要设置的参数
        word_embedding_fp = argv[0]  # word embedding 路径
        feature_name = argv[1]  # 特征名字
        WordEmbedding.len_vec = int(argv[2])  # word embedding 维度
        WordEmbedding.to_lower = bool(argv[3])  # 是否需要转化为小写

        # 加载 word embedding 词典
        WordEmbedding.we_dict = WordEmbedding.load_word_embedding(word_embedding_fp)
        LogUtil.log('INFO', 'load word embedding dict done')

        # 计算IDF词表
        train_qid2q_fp = '%s/train_qid2question.csv' % cf.get('DEFAULT', 'devel_pt')
        train_qid2q = pd.read_csv(train_qid2q_fp).fillna(value="")
        WordEmbedding.init_idf(train_qid2q)

        # 加载数据文件
        train_data = pd.read_csv('%s/train.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")
        test_data = pd.read_csv('%s/test_with_qid.csv' % cf.get('DEFAULT', 'devel_pt')).fillna(value="")

        # 特征存储路径
        feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
        train_feature_fp = '%s/%s.train.smat' % (feature_pt, feature_name)
        test_feature_fp = '%s/%s.test.smat' % (feature_pt, feature_name)

        # 抽取特征：train.csv
        train_features = train_data.apply(WordEmbedding.extract_row_tfidf_dis, axis=1, raw=True)
        LogUtil.log('INFO', 'extract train features (tfidf) done')
        test_features = test_data.apply(WordEmbedding.extract_row_tfidf_dis, axis=1, raw=True)
        LogUtil.log('INFO', 'extract test features (tfidf) done')
        # 抽取特征: test.csv
        Feature.save_dataframe(train_features, train_feature_fp)
        LogUtil.log('INFO', 'save train features (tfidf) done')
        Feature.save_dataframe(test_features, test_feature_fp)
        LogUtil.log('INFO', 'save test features (tfidf) done')

    @staticmethod
    def extract_row_ave_vec(row):
        """
        按行抽取特征
        :param row:
        :return:
        """
        q1_words = str(row['question1']).strip().split() if WordEmbedding.to_lower else str(
            row['question1']).lower().strip().split()
        q2_words = str(row['question2']).strip().split() if WordEmbedding.to_lower else str(
            row['question2']).lower().strip().split()

        q1_vec = np.array(WordEmbedding.len_vec * [0.])
        q2_vec = np.array(WordEmbedding.len_vec * [0.])

        for word in q1_words:
            if word in WordEmbedding.we_dict:
                q1_vec = q1_vec + WordEmbedding.we_dict[word]
        for word in q2_words:
            if word in WordEmbedding.we_dict:
                q2_vec = q2_vec + WordEmbedding.we_dict[word]

        return list(q1_vec) + list(q2_vec)

    @staticmethod
    def extract_ave_vec(cf, argv):
        # 运行需要设置的参数
        word_embedding_fp = argv[0]  # word embedding 路径
        feature_name = argv[1]  # 特征名字
        WordEmbedding.len_vec = int(argv[2])  # word embedding 维度
        WordEmbedding.to_lower = bool(argv[3])  # 是否需要转化为小写

        # 加载 word embedding 词典
        WordEmbedding.we_dict = WordEmbedding.load_word_embedding(word_embedding_fp)
        LogUtil.log('INFO', 'load word embedding dict done')

        train_data = pd.read_csv('%s/train.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")
        test_data = pd.read_csv('%s/test_with_qid.csv' % cf.get('DEFAULT', 'devel_pt')).fillna(value="")

        feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
        train_feature_fp = '%s/%s.train.smat' % (feature_pt, feature_name)
        test_feature_fp = '%s/%s.test.smat' % (feature_pt, feature_name)

        train_features = train_data.apply(WordEmbedding.extract_row_ave_vec, axis=1, raw=True)
        LogUtil.log('INFO', 'extract train features (ave_vec) done')
        test_features = test_data.apply(WordEmbedding.extract_row_ave_vec, axis=1, raw=True)
        LogUtil.log('INFO', 'extract test features (ave_vec) done')

        Feature.save_dataframe(train_features, train_feature_fp)
        LogUtil.log('INFO', 'save train features done')
        Feature.save_dataframe(test_features, test_feature_fp)
        LogUtil.log('INFO', 'save test features done')

    @staticmethod
    def extract_row_tfidf_vec(row):
        """
        按行抽取特征
        :param row:
        :return:
        """
        q1_words = str(row['question1']).strip().split() if WordEmbedding.to_lower else str(
            row['question1']).lower().strip().split()
        q2_words = str(row['question2']).strip().split() if WordEmbedding.to_lower else str(
            row['question2']).lower().strip().split()

        q1_vec = np.array(WordEmbedding.len_vec * [0.])
        q2_vec = np.array(WordEmbedding.len_vec * [0.])

        q1_words_cnt = {}
        q2_words_cnt = {}
        for word in q1_words:
            q1_words_cnt[word] = q1_words_cnt.get(word, 0.) + 1.
        for word in q2_words:
            q2_words_cnt[word] = q2_words_cnt.get(word, 0.) + 1.

        for word in q1_words_cnt:
            if word in WordEmbedding.we_dict:
                q1_vec = q1_vec + WordEmbedding.idf.get(word, 0.) * q1_words_cnt[word] * WordEmbedding.we_dict[word]
        for word in q2_words_cnt:
            if word in WordEmbedding.we_dict:
                q2_vec = q2_vec + WordEmbedding.idf.get(word, 0.) * q2_words_cnt[word] * WordEmbedding.we_dict[word]

        return list(q1_vec) + list(q2_vec)

    @staticmethod
    def extract_tfidf_vec(cf, argv):
        # 运行需要设置的参数
        word_embedding_fp = argv[0]  # word embedding 路径
        feature_name = argv[1]  # 特征名字
        WordEmbedding.len_vec = int(argv[2])  # word embedding 维度
        WordEmbedding.to_lower = bool(argv[3])  # 是否需要转化为小写

        # 加载 word embedding 词典
        WordEmbedding.we_dict = WordEmbedding.load_word_embedding(word_embedding_fp)
        LogUtil.log('INFO', 'load word embedding dict done')

        # 计算IDF词表
        train_qid2q_fp = '%s/train_qid2question.csv' % cf.get('DEFAULT', 'devel_pt')
        train_qid2q = pd.read_csv(train_qid2q_fp).fillna(value="")
        WordEmbedding.init_idf(train_qid2q)

        # 加载数据文件
        train_data = pd.read_csv('%s/train.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")
        test_data = pd.read_csv('%s/test_with_qid.csv' % cf.get('DEFAULT', 'devel_pt')).fillna(value="")

        # 特征存储路径
        feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
        train_feature_fp = '%s/%s.train.smat' % (feature_pt, feature_name)
        test_feature_fp = '%s/%s.test.smat' % (feature_pt, feature_name)

        # 抽取特征：train.csv
        train_features = train_data.apply(WordEmbedding.extract_row_tfidf_vec, axis=1, raw=True)
        LogUtil.log('INFO', 'extract train features (tfidf_vec) done')
        test_features = test_data.apply(WordEmbedding.extract_row_tfidf_vec, axis=1, raw=True)
        LogUtil.log('INFO', 'extract test features (tfidf_vec) done')
        # 抽取特征: test.csv
        Feature.save_dataframe(train_features, train_feature_fp)
        LogUtil.log('INFO', 'save train features (tfidf_vec) done')
        Feature.save_dataframe(test_features, test_feature_fp)
        LogUtil.log('INFO', 'save test features (tfidf_vec) done')

    @staticmethod
    def run(argv):
        """
        运行所有 Word Embedding 特征抽取器
        :return:
        """
        # 读取配置文件
        cf = ConfigParser.ConfigParser()
        cf.read("../conf/python.conf")

        # 运行抽取器
        cmd = argv[0]
        if 'extract_ave_dis' == cmd:
            WordEmbedding.extract_ave_dis(cf, argv[1:])
        elif 'extract_tfidf_dis' == cmd:
            WordEmbedding.extract_tfidf_dis(cf, argv[1:])
        elif 'extract_ave_vec' == cmd:
            WordEmbedding.extract_ave_vec(cf, argv[1:])
        elif 'extract_tfidf_vec' == cmd:
            WordEmbedding.extract_tfidf_vec(cf, argv[1:])


class ID(object):

    question2id = {}

    @staticmethod
    def extract_row_id(row):
        """
        按行抽取特征
        :param row:
        :return:
        """
        q1 = str(row['question1']).strip()
        q2 = str(row['question2']).strip()

        v = max(ID.question2id.setdefault(q1, len(ID.question2id)), ID.question2id.setdefault(q2, len(ID.question2id)))

        return [v]


    @staticmethod
    def extract_id(cf, argv):
        # 设置参数
        feature_name = 'id'

        # 加载数据文件
        train_data = pd.read_csv('%s/train.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")
        test_data = pd.read_csv('%s/test_with_qid.csv' % cf.get('DEFAULT', 'devel_pt')).fillna(value="")

        # 特征存储路径
        feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
        train_feature_fp = '%s/%s.train.smat' % (feature_pt, feature_name)
        test_feature_fp = '%s/%s.test.smat' % (feature_pt, feature_name)

        # 抽取特征：train.csv
        train_features = train_data.apply(ID.extract_row_id, axis=1, raw=True)
        LogUtil.log('INFO', 'extract train features (%s) done' % feature_name)
        test_features = test_data.apply(ID.extract_row_id, axis=1, raw=True)
        LogUtil.log('INFO', 'extract test features (%s) done' % feature_name)
        # 抽取特征: test.csv
        Feature.save_dataframe(train_features, train_feature_fp)
        LogUtil.log('INFO', 'save train features (%s) done' % feature_name)
        Feature.save_dataframe(test_features, test_feature_fp)
        LogUtil.log('INFO', 'save test features (%s) done' % feature_name)

    @staticmethod
    def run(argv):
        # 读取配置文件
        cf = ConfigParser.ConfigParser()
        cf.read("../conf/python.conf")

        # 运行抽取器
        ID.extract_id(cf, argv)


class POSTag(object):

    def __init__(self):
        pass

    postag = {}

    @staticmethod
    def init_row_postag(row):
        q1_postag = json.loads(row['question1_postag'])
        for sentence in q1_postag:
            for kv in sentence:
                POSTag.postag.setdefault(kv[1], len(POSTag.postag))

        q2_postag = json.loads(row['question2_postag'])
        for sentence in q2_postag:
            for kv in sentence:
                POSTag.postag.setdefault(kv[1], len(POSTag.postag))


    @staticmethod
    def extract_row_postag_cnt(row):
        q1_vec = len(POSTag.postag) * [0]
        q1_postag = json.loads(row['question1_postag'])
        for s in q1_postag:
            for kv in s:
                id = POSTag.postag[kv[1]]
                q1_vec[id] += 1

        q2_vec = len(POSTag.postag) * [0]
        q2_postag = json.loads(row['question2_postag'])
        for s in q2_postag:
            for kv in s:
                id = POSTag.postag[kv[1]]
                q2_vec[id] += 1

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

    @staticmethod
    def extract_postag_cnt(cf, argv):
        # 设置参数
        feature_name = 'postag_cnt'

        # 加载数据文件
        train_data = pd.read_csv('%s/train_postag.csv' % cf.get('DEFAULT', 'devel_pt')).fillna(value="")
        test_data = pd.read_csv('%s/test_postag.csv' % cf.get('DEFAULT', 'devel_pt')).fillna(value="")

        # 特征存储路径
        feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
        train_feature_fp = '%s/%s.train.smat' % (feature_pt, feature_name)
        test_feature_fp = '%s/%s.test.smat' % (feature_pt, feature_name)

        # 初始化
        POSTag.postag = {}
        train_data.apply(POSTag.init_row_postag, axis=1, raw=True)
        test_data.apply(POSTag.init_row_postag, axis=1, raw=True)
        LogUtil.log('INFO', 'len(postag)=%d, postag=%s' % (len(POSTag.postag), str(POSTag.postag)))

        # 抽取特征：train.csv
        train_features = train_data.apply(POSTag.extract_row_postag_cnt, axis=1, raw=True)
        LogUtil.log('INFO', 'extract train features (%s) done' % feature_name)
        test_features = test_data.apply(POSTag.extract_row_postag_cnt, axis=1, raw=True)
        LogUtil.log('INFO', 'extract test features (%s) done' % feature_name)
        # 抽取特征: test.csv
        Feature.save_dataframe(train_features, train_feature_fp)
        LogUtil.log('INFO', 'save train features (%s) done' % feature_name)
        Feature.save_dataframe(test_features, test_feature_fp)
        LogUtil.log('INFO', 'save test features (%s) done' % feature_name)


    @staticmethod
    def run(argv):
        # 运行抽取器
        POSTag.extract_postag_cnt(cf, argv)


class DulNum(object):

    dul_num = {}

    def __init__(self):
        pass

    @staticmethod
    def init_dul_num(train_data, test_data):
        for index, row in train_data.iterrows():
            q1 = str(row.question1).strip()
            q2 = str(row.question2).strip()
            DulNum.dul_num[q1] = DulNum.dul_num.get(q1, 0) + 1
            if q1 != q2:
                DulNum.dul_num[q2] = DulNum.dul_num.get(q2, 0) + 1

        for index, row in test_data.iterrows():
            q1 = str(row.question1).strip()
            q2 = str(row.question2).strip()
            DulNum.dul_num[q1] = DulNum.dul_num.get(q1, 0) + 1
            if q1 != q2:
                DulNum.dul_num[q2] = DulNum.dul_num.get(q2, 0) + 1

    @staticmethod
    def extract_row_dul_num(row):
        q1 = str(row['question1']).strip()
        q2 = str(row['question2']).strip()

        dn1 = DulNum.dul_num[q1]
        dn2 = DulNum.dul_num[q2]

        return [dn1, dn2, max(dn1, dn2), min(dn1, dn2)]

    @staticmethod
    def extract_dul_num(cf, argv):
        # 设置参数
        feature_name = 'dul_num'

        # 加载数据文件
        train_data = pd.read_csv('%s/train_postag.csv' % cf.get('DEFAULT', 'devel_pt')).fillna(value="")
        test_data = pd.read_csv('%s/test_postag.csv' % cf.get('DEFAULT', 'devel_pt')).fillna(value="")

        # 特征存储路径
        feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
        train_feature_fp = '%s/%s.train.smat' % (feature_pt, feature_name)
        test_feature_fp = '%s/%s.test.smat' % (feature_pt, feature_name)

        # 初始化
        DulNum.init_dul_num(train_data, test_data)
        LogUtil.log('INFO', 'len(dul_num)=%d' % (len(DulNum.dul_num)))

        # 抽取特征：train.csv
        train_features = train_data.apply(DulNum.extract_row_dul_num, axis=1, raw=True)
        LogUtil.log('INFO', 'extract train features (%s) done' % feature_name)
        test_features = test_data.apply(DulNum.extract_row_dul_num, axis=1, raw=True)
        LogUtil.log('INFO', 'extract test features (%s) done' % feature_name)
        # 抽取特征: test.csv
        Feature.save_dataframe(train_features, train_feature_fp)
        LogUtil.log('INFO', 'save train features (%s) done' % feature_name)
        Feature.save_dataframe(test_features, test_feature_fp)
        LogUtil.log('INFO', 'save test features (%s) done' % feature_name)


    @staticmethod
    def extract_row_dul_num_ratio(row):
        q1 = str(row['question1']).strip()
        q2 = str(row['question2']).strip()

        dn1 = DulNum.dul_num[q1]
        dn2 = DulNum.dul_num[q2]

        ratio = (1.0 * dn1 / dn2) if dn2 > 1e-12 else 0.0

        return [ratio]

    @staticmethod
    def extract_dul_num_ratio(cf, argv):
        # 设置参数
        feature_name = 'dul_num_ratio'

        # 加载数据文件
        train_data = pd.read_csv('%s/train.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")
        test_data = pd.read_csv('%s/test_with_qid.csv' % cf.get('DEFAULT', 'devel_pt')).fillna(value="")

        # 特征存储路径
        feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
        train_feature_fp = '%s/%s.train.smat' % (feature_pt, feature_name)
        test_feature_fp = '%s/%s.test.smat' % (feature_pt, feature_name)

        # 初始化
        DulNum.init_dul_num(train_data, test_data)
        LogUtil.log('INFO', 'len(dul_num)=%d' % (len(DulNum.dul_num)))

        # 抽取特征：train.csv
        train_features = train_data.apply(DulNum.extract_row_dul_num_ratio, axis=1, raw=True)
        LogUtil.log('INFO', 'extract train features (%s) done' % feature_name)
        test_features = test_data.apply(DulNum.extract_row_dul_num_ratio, axis=1, raw=True)
        LogUtil.log('INFO', 'extract test features (%s) done' % feature_name)
        # 抽取特征: test.csv
        Feature.save_dataframe(train_features, train_feature_fp)
        LogUtil.log('INFO', 'save train features (%s) done' % feature_name)
        Feature.save_dataframe(test_features, test_feature_fp)
        LogUtil.log('INFO', 'save test features (%s) done' % feature_name)

    @staticmethod
    def run(argv):
        cmd = argv[0]

        # 运行抽取器
        if 'dul_num' == cmd:
            DulNum.extract_dul_num(cf, argv[1:])
        elif 'dul_num_ratio' == cmd:
            DulNum.extract_dul_num_ratio(cf, argv[1:])


class MathTag(object):

    def __init__(self):
        pass

    @staticmethod
    def extract_row_math_tag(row):
        q1 = str(row['question1']).strip()
        q2 = str(row['question2']).strip()

        q1_cnt = q1.count('[math]')
        q2_cnt = q2.count('[math]')

        pair_and = int((0 < q1_cnt) and (0 < q2_cnt))
        pair_or = int((0 < q1_cnt) or (0 < q2_cnt))

        return [q1_cnt, q2_cnt, pair_and, pair_or]


    @staticmethod
    def extract_math_tag(cf, argv):
        # 设置参数
        feature_name = 'math_tag'

        # 加载数据文件
        train_data = pd.read_csv('%s/train.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")
        test_data = pd.read_csv('%s/test_with_qid.csv' % cf.get('DEFAULT', 'devel_pt')).fillna(value="")

        # 特征存储路径
        feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
        train_feature_fp = '%s/%s.train.smat' % (feature_pt, feature_name)
        test_feature_fp = '%s/%s.test.smat' % (feature_pt, feature_name)

        # 抽取特征：train.csv
        train_features = train_data.apply(MathTag.extract_row_math_tag, axis=1, raw=True)
        LogUtil.log('INFO', 'extract train features (%s) done' % feature_name)
        test_features = test_data.apply(MathTag.extract_row_math_tag, axis=1, raw=True)
        LogUtil.log('INFO', 'extract test features (%s) done' % feature_name)
        # 抽取特征: test.csv
        Feature.save_dataframe(train_features, train_feature_fp)
        LogUtil.log('INFO', 'save train features (%s) done' % feature_name)
        Feature.save_dataframe(test_features, test_feature_fp)
        LogUtil.log('INFO', 'save test features (%s) done' % feature_name)


    @staticmethod
    def run(argv):
        # 运行抽取器
        MathTag.extract_math_tag(cf, argv)


class BTMVecCosSimDis(object):
    def __init__(self):
        pass

    @staticmethod
    def load_btm_vec_cos_sim_dis(cf):
        features = []
        new_line = True
        for index in range(10):
            f = open(cf.get('DEFAULT', 'devel_pt') + '/btm_train_wordtoken_100_50.fs.0%d' % index)
            for line in f:
                vec = [float(s) for s in line.strip().split()]
                if new_line:
                    new_line = False
                    features.append(vec[0:49])
                else:
                    new_line = True
                    features[len(features) - 1].extend(vec[0:49])
            f.close()
        return features

    @staticmethod
    def extract_btm_vec_cos_sim_dis(cf, argv):
        # 设置参数
        feature_name = 'btm_train_wordtoken_100_50_dis_98'

        # 加载数据文件
        train_data = pd.read_csv('%s/train.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")
        test_data = pd.read_csv('%s/test_with_qid.csv' % cf.get('DEFAULT', 'devel_pt')).fillna(value="")

        # 初始化特征向量表
        features = BTMVecCosSimDis.load_btm_vec_cos_sim_dis(cf)

        # 特征存储路径
        feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
        train_feature_fp = '%s/%s.train.smat' % (feature_pt, feature_name)
        test_feature_fp = '%s/%s.test.smat' % (feature_pt, feature_name)

        # 抽取特征：train.csv
        train_features = features[0 : len(train_data)]
        LogUtil.log('INFO', 'extract train features (%s) done' % feature_name)
        Feature.save_dataframe(train_features, train_feature_fp)
        LogUtil.log('INFO', 'save train features (%s) done' % feature_name)

        test_features = features[len(train_data) : len(train_data) + len(test_data)]
        # 增加一个自身向量的修正
        # for i in range(len(test_features)):
        #     test_features[i][49] += 1.0
        #     test_features[i][99] += 1.0
        LogUtil.log('INFO', 'extract test features (%s) done' % feature_name)
        Feature.save_dataframe(test_features, test_feature_fp)
        LogUtil.log('INFO', 'save test features (%s) done' % feature_name)

    @staticmethod
    def plot_btm_vec_cos_sim_dis(cf, argv):
        # 设置参数
        feature_name = 'btm_vec_cos_sim_dis'

        # 加载数据文件
        train_data = pd.read_csv('%s/train.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")

        # 加载特征
        train_features = pd.DataFrame(Feature.load('%s/%s.train.smat' % (cf.get('DEFAULT', 'feature_question_pair_pt'), feature_name)).toarray())

        # 统计
        # 负例
        neg_train_features = train_features[train_data['is_duplicate'] == 0].apply(lambda x: x[11], axis=1, raw=True)
        LogUtil.log("INFO", 'neg: mean=%.2f, std=%.2f, max=%.2f, min=%.2f' % (
            neg_train_features.mean(), neg_train_features.std(), neg_train_features.max(), neg_train_features.min()))
        # print neg_train_features
        # 正例
        pos_train_features = train_features[train_data['is_duplicate'] == 1].apply(lambda x: x[11], axis=1, raw=True)
        LogUtil.log("INFO", 'pos: mean=%.2f, std=%.2f, max=%.2f, min=%.2f' % (
            pos_train_features.mean(), pos_train_features.std(), pos_train_features.max(), pos_train_features.min()))

        # 绘图
        plt.figure(figsize=(15, 5))
        plt.xlim([0, 1])
        plt.hist(neg_train_features, bins=100, normed=False, label='Not Duplicate', edgecolor='None')
        plt.hist(pos_train_features, bins=100, normed=False, alpha=0.7, label='Duplicate', edgecolor='None')
        plt.legend()
        plt.title('Label distribution over %s' % feature_name, fontsize=15)
        plt.xlabel(feature_name, fontsize=15)
        # plt.show()


    @staticmethod
    def run(argv):
        cmd = argv[0]

        # 运行抽取器
        if 'extract_btm_vec_cos_sim_dis' == cmd:
            BTMVecCosSimDis.extract_btm_vec_cos_sim_dis(cf, argv[1:])
        elif 'plot_btm_vec_cos_sim_dis' == cmd:
            BTMVecCosSimDis.plot_btm_vec_cos_sim_dis(cf, argv[1:])


class PowerfulWordV2(object):
    snowball_stemmer = SnowballStemmer('english')
    """
    寻找最优影响力的词
    """
    dside_word_power = []
    oside_word_power = []
    aside_word_power = []
    word_power_dict = []

    @staticmethod
    def cal_word_power(train_data):
        """
        计算数据中词语的影响力，格式如下：
            词语 --> [0. 出现语句对数量，1. 出现语句对比例，2. 正确语句对比例，3. 单侧语句对比例，4. 单侧语句对正确比例，5. 双侧语句对比例，6. 双侧语句对正确比例]
        :param data: 训练数据
        :return: 影响力词典
        """
        words_power = {}
        for index, row in train_data.iterrows():
            label = int(row['is_duplicate'])
            q1_words = [PowerfulWordV2.snowball_stemmer.stem(word).encode('utf-8') for word in nltk.word_tokenize(Preprocessor.clean_text(str(row['question1']).decode('utf-8')))]
            q2_words = [PowerfulWordV2.snowball_stemmer.stem(word).encode('utf-8') for word in nltk.word_tokenize(Preprocessor.clean_text(str(row['question2']).decode('utf-8')))]
            all_words = set(q1_words + q2_words)
            q1_words = set(q1_words)
            q2_words = set(q2_words)
            for word in all_words:
                if word not in words_power:
                    words_power[word] = [0. for i in range(7)]
                # 计算出现语句对数量
                words_power[word][0] += 1.
                words_power[word][1] += 1.

                if ((word in q1_words) and (word not in q2_words)) or ((word not in q1_words) and (word in q2_words)):
                    # 计算单侧语句数量
                    words_power[word][3] += 1.
                    if 0 == label:
                        # 计算正确语句对数量
                        words_power[word][2] += 1.
                        # 计算单侧语句正确比例
                        words_power[word][4] += 1.
                if (word in q1_words) and (word in q2_words):
                    # 计算双侧语句数量
                    words_power[word][5] += 1.
                    if 1 == label:
                        # 计算正确语句对数量
                        words_power[word][2] += 1.
                        # 计算双侧语句正确比例
                        words_power[word][6] += 1.
        for word in words_power:
            # 计算出现语句对比例
            words_power[word][1] /= len(train_data)
            # 计算正确语句对比例
            words_power[word][2] /= words_power[word][0]
            # 计算单侧语句对正确比例
            if words_power[word][3] > 1e-6:
                words_power[word][4] /= words_power[word][3]
            # 计算单侧语句对比例
            words_power[word][3] /= words_power[word][0]
            # 计算双侧语句对正确比例
            if words_power[word][5] > 1e-6:
                words_power[word][6] /= words_power[word][5]
            # 计算双侧语句对比例
            words_power[word][5] /= words_power[word][0]
        sorted_words_power = sorted(words_power.iteritems(), key=lambda d: d[1][0], reverse=True)
        LogUtil.log("INFO", "power words calculation done, len(words_power)=%d" % len(sorted_words_power))
        return sorted_words_power

    @staticmethod
    def init_dside_word_power(words_power):
        """
        初始化双边影响力词表
        :param words_power: 影响力词表
        :return: NONE
        """
        PowerfulWordV2.dside_word_power = []
        # 在双边pair中最少出现的次数
        num_least = 300
        words_power = filter(lambda x: x[1][0] * x[1][5] >= num_least, words_power)
        # 按照双侧语句对正确比例排序
        sorted_words_power = sorted(words_power, key=lambda d: d[1][6], reverse=True)
        # 双侧正确比例阈值
        dside_corate_rate = 0.7
        PowerfulWordV2.dside_word_power.extend(
            map(lambda x: x[0], filter(lambda x: x[1][6] >= dside_corate_rate, sorted_words_power)))
        LogUtil.log('INFO', 'Double side power words(%d): %s' % (
            len(PowerfulWordV2.dside_word_power), str(PowerfulWordV2.dside_word_power)))

    @staticmethod
    def init_oside_word_power(words_power):
        """
        初始化单边影响力词表
        :param words_power:
        :return:
        """
        PowerfulWordV2.oside_word_power = []
        # 在单边pair中最少出现次数
        num_least = 500
        words_power = filter(lambda x: x[1][0] * x[1][3] >= num_least, words_power)
        # 单边正确比例阈值
        oside_corate_rate = 0.9
        PowerfulWordV2.oside_word_power.extend(
            map(lambda x: x[0], filter(lambda x: x[1][4] >= oside_corate_rate, words_power)))
        LogUtil.log('INFO', 'One side power words(%d): %s' % (
            len(PowerfulWordV2.oside_word_power), str(PowerfulWordV2.oside_word_power)))

    @staticmethod
    def load_word_power(fp):
        """
        加载影响力词表
        :param fp: 影响力词表路径
        :return: 影响力词表
        """
        words_power = []
        f = open(fp, 'r')
        for line in f:
            subs = line.split('\t')
            word = subs[0]
            stats = [float(num) for num in subs[1:]]
            words_power.append((word, stats))
        f.close()
        return words_power

    @staticmethod
    def save_word_power(words_power, fp):
        """
        存储影响力词表
        :param words_power: 影响力词表
        :param fp: 存储路径
        :return: NONE
        """
        f = open(fp, 'w')
        for ele in words_power:
            f.write("%s" % ele[0])
            for num in ele[1]:
                f.write("\t%.6f" % num)
            f.write("\n")
        f.close()

    @staticmethod
    def generate_powerful_word(cf, argv):
        # 加载train.csv文件
        train_data = pd.read_csv('%s/train.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")

        # 影响力词表路径
        words_power_fp = '%s/words_power_v2.train.txt' % (cf.get('DEFAULT', 'feature_stat_pt'))

        # 生成影响力词表
        words_power = PowerfulWordV2.cal_word_power(train_data)

        # 存储影响力词表
        PowerfulWordV2.save_word_power(words_power, words_power_fp)

    @staticmethod
    def extract_row_dside_word_power(row):
        """
        针对一个Pair抽取特征：是否包含双边影响力词表
        :param row: 一个Pair实例
        :return: Tags
        """
        tags = []
        q1_words = [PowerfulWordV2.snowball_stemmer.stem(word).encode('utf-8') for word in
                    nltk.word_tokenize(Preprocessor.clean_text(str(row['question1']).decode('utf-8')))]
        q2_words = [PowerfulWordV2.snowball_stemmer.stem(word).encode('utf-8') for word in
                    nltk.word_tokenize(Preprocessor.clean_text(str(row['question2']).decode('utf-8')))]
        for word in PowerfulWordV2.dside_word_power:
            if (word in q1_words) and (word in q2_words):
                tags.append(1.0)
            else:
                tags.append(0.0)
        return tags

    @staticmethod
    def extract_row_oside_word_power(row):
        """
        针对一个Pair抽取特征：是否包含单边影响力词表
        :param row: 一个Pair实例
        :return: Tags
        """
        tags = []
        q1_words = [PowerfulWordV2.snowball_stemmer.stem(word).encode('utf-8') for word in
                    nltk.word_tokenize(Preprocessor.clean_text(str(row['question1']).decode('utf-8')))]
        q2_words = [PowerfulWordV2.snowball_stemmer.stem(word).encode('utf-8') for word in
                    nltk.word_tokenize(Preprocessor.clean_text(str(row['question2']).decode('utf-8')))]
        for word in PowerfulWordV2.oside_word_power:
            if (word in q1_words) and (word in q2_words):
                tags.append(1.0)
            else:
                tags.append(0.0)
        return tags

    @staticmethod
    def extract_dside_word_power_v2(cf, argv):
        # 设置参数
        feature_name = 'dside_word_power_v2'

        # 加载数据文件
        train_data = pd.read_csv('%s/train.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")
        test_data = pd.read_csv('%s/test_with_qid.csv' % cf.get('DEFAULT', 'devel_pt')).fillna(value="")

        # 特征存储路径
        feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
        train_feature_fp = '%s/%s.train.smat' % (feature_pt, feature_name)
        test_feature_fp = '%s/%s.test.smat' % (feature_pt, feature_name)

        # 加载词表
        words_power_fp = '%s/words_power_v2.train.txt' % (cf.get('DEFAULT', 'feature_stat_pt'))
        words_power = PowerfulWordV2.load_word_power(words_power_fp)

        PowerfulWordV2.init_dside_word_power(words_power)


        # 抽取特征：train.csv
        train_features = train_data.apply(PowerfulWordV2.extract_row_dside_word_power, axis=1, raw=True)
        LogUtil.log('INFO', 'extract train features (%s) done' % feature_name)
        test_features = test_data.apply(PowerfulWordV2.extract_row_dside_word_power, axis=1, raw=True)
        LogUtil.log('INFO', 'extract test features (%s) done' % feature_name)
        # 抽取特征: test.csv
        Feature.save_dataframe(train_features, train_feature_fp)
        LogUtil.log('INFO', 'save train features (%s) done' % feature_name)
        Feature.save_dataframe(test_features, test_feature_fp)
        LogUtil.log('INFO', 'save test features (%s) done' % feature_name)

    @staticmethod
    def extract_oside_word_power_v2(cf, argv):
        # 设置参数
        feature_name = 'oside_word_power_v2'

        # 加载数据文件
        train_data = pd.read_csv('%s/train.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")
        test_data = pd.read_csv('%s/test_with_qid.csv' % cf.get('DEFAULT', 'devel_pt')).fillna(value="")

        # 特征存储路径
        feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
        train_feature_fp = '%s/%s.train.smat' % (feature_pt, feature_name)
        test_feature_fp = '%s/%s.test.smat' % (feature_pt, feature_name)

        # 加载词表
        words_power_fp = '%s/words_power_v2.train.txt' % (cf.get('DEFAULT', 'feature_stat_pt'))
        words_power = PowerfulWordV2.load_word_power(words_power_fp)

        PowerfulWordV2.init_oside_word_power(words_power)

        # 抽取特征：train.csv
        train_features = train_data.apply(PowerfulWordV2.extract_row_oside_word_power, axis=1, raw=True)
        LogUtil.log('INFO', 'extract train features (%s) done' % feature_name)
        Feature.save_dataframe(train_features, train_feature_fp)
        LogUtil.log('INFO', 'save train features (%s) done' % feature_name)

        test_features = test_data.apply(PowerfulWordV2.extract_row_oside_word_power, axis=1, raw=True)
        LogUtil.log('INFO', 'extract test features (%s) done' % feature_name)
        Feature.save_dataframe(test_features, test_feature_fp)
        LogUtil.log('INFO', 'save test features (%s) done' % feature_name)

    @staticmethod
    def run(cf, argv):
        cmd = argv[0]

        if 'generate_powerful_word' == cmd:
            PowerfulWordV2.generate_powerful_word(cf, argv[1:])
        elif 'extract_dside_word_power_v2' == cmd:
            PowerfulWordV2.extract_dside_word_power_v2(cf, argv[1:])
        elif 'extract_oside_word_power_v2' == cmd:
            PowerfulWordV2.extract_oside_word_power_v2(cf, argv[1:])


class Graph(object):
    G = None
    q2id = None
    p2weight = None
    pr = None
    hits_h = None
    hits_a = None
    counter = 0

    def __init__(self):
        pass

    @staticmethod
    def init_graph(cf, argv):
        Graph.q2id = {}

        fin = csv.reader(open('%s/train.csv' % cf.get('DEFAULT', 'origin_pt')))
        fin.next()
        fout = open('%s/graph_question2id.train.txt' % cf.get('DEFAULT', 'devel_pt'), 'w')
        for p in fin:
            q1 = str(p[3]).strip()
            q2 = str(p[4]).strip()
            label = p[5]
            if q1 not in Graph.q2id:
                Graph.q2id[q1] = len(Graph.q2id)
            if q2 not in Graph.q2id:
                Graph.q2id[q2] = len(Graph.q2id)
            print >> fout, Graph.q2id[q1], Graph.q2id[q2], label
        LogUtil.log('INFO', 'len(questions)=%d' % len(Graph.q2id))
        fout.close()

        fin = csv.reader(open('%s/test.csv' % cf.get('DEFAULT', 'origin_pt')))
        fin.next()
        fout = open('%s/graph_question2id.test.txt' % cf.get('DEFAULT', 'devel_pt'), 'w')
        for p in fin:
            q1 = str(p[1]).strip()
            q2 = str(p[2]).strip()
            if q1 not in Graph.q2id:
                Graph.q2id[q1] = len(Graph.q2id)
            if q2 not in Graph.q2id:
                Graph.q2id[q2] = len(Graph.q2id)
            print >> fout, Graph.q2id[q1], Graph.q2id[q2]
        LogUtil.log('INFO', 'len(questions)=%d' % len(Graph.q2id))
        fout.close()

        Graph.G = nx.Graph()
        for line in open('%s/graph_question2id.train.txt' % cf.get('DEFAULT', 'devel_pt')):
            head, tail, label = [int(x) for x in line.split()]
            Graph.G.add_edge(head, tail, weight=label + 1)
        for line in open('%s/graph_question2id.test.txt' % cf.get('DEFAULT', 'devel_pt')):
            head, tail = [int(x) for x in line.split()]
            Graph.G.add_edge(head, tail, weight=0)
        LogUtil.log('INFO', 'Graph constructed.')

        # ccs = nx.connected_components(G)
        # Graph.connected_components = []
        # Graph.p2cc = {}
        # for cc in ccs:
        #     for p in cc:
        #         if p in Graph.p2cc:
        #             LogUtil.log('WARNING', '%d already in p2cc(=%d)' % (p, Graph.p2cc[p]))
        #         else:
        #             Graph.p2cc[p] = len(Graph.connected_components)
        #     Graph.connected_components.append(cc)
        #
        # LogUtil.log('INFO', 'len(Graph.connected_components)=%d' % len(Graph.connected_components))
        # LogUtil.log('INFO', 'len(Graph.p2cc)=%d' % len(Graph.p2cc))

    @staticmethod
    def init_graph_with_weight(cf, weight_featue_name, weight_feature_id,  reverse=False):
        Graph.q2id = {}
        Graph.p2weight = {}
        Graph.G = nx.Graph()

        train_wfs_fs = Feature.load(
            '%s/%s.train.smat' % (cf.get('DEFAULT', 'feature_question_pair_pt'), weight_featue_name)).toarray()
        test_wfs_fs = Feature.load(
            '%s/%s.test.smat' % (cf.get('DEFAULT', 'feature_question_pair_pt'), weight_featue_name)).toarray()

        if 'True' == reverse:
            LogUtil.log('INFO', 'will reverse')
            for index in range(len(train_wfs_fs)):
                train_wfs_fs[index][weight_feature_id] = 1. - train_wfs_fs[index][weight_feature_id]
            for index in range(len(test_wfs_fs)):
                test_wfs_fs[index][weight_feature_id] = 1. - test_wfs_fs[index][weight_feature_id]

        fin = csv.reader(open('%s/train.csv' % cf.get('DEFAULT', 'origin_pt')))
        fin.next()
        # fout = open('%s/graph_question2id.train.txt' % cf.get('DEFAULT', 'devel_pt'), 'w')
        index = 0
        for p in fin:
            q1 = str(p[3]).strip()
            q2 = str(p[4]).strip()
            label = p[5]
            weight = train_wfs_fs[index][weight_feature_id]
            if q1 not in Graph.q2id:
                Graph.q2id[q1] = len(Graph.q2id)
            if q2 not in Graph.q2id:
                Graph.q2id[q2] = len(Graph.q2id)
            # print >> fout, Graph.q2id[q1], Graph.q2id[q2], label, weight
            Graph.G.add_edge(Graph.q2id[q1], Graph.q2id[q2], weight=weight)
            Graph.p2weight[(Graph.q2id[q1], Graph.q2id[q2])] = weight
            Graph.p2weight[(Graph.q2id[q2], Graph.q2id[q1])] = weight
            index += 1
        LogUtil.log('INFO', 'len(questions)=%d' % len(Graph.q2id))
        # fout.close()

        fin = csv.reader(open('%s/test.csv' % cf.get('DEFAULT', 'origin_pt')))
        fin.next()
        # fout = open('%s/graph_question2id.test.txt' % cf.get('DEFAULT', 'devel_pt'), 'w')
        index  = 0
        for p in fin:
            q1 = str(p[1]).strip()
            q2 = str(p[2]).strip()
            weight = test_wfs_fs[index][weight_feature_id]
            if q1 not in Graph.q2id:
                Graph.q2id[q1] = len(Graph.q2id)
            if q2 not in Graph.q2id:
                Graph.q2id[q2] = len(Graph.q2id)
            # print >> fout, Graph.q2id[q1], Graph.q2id[q2], weight
            Graph.G.add_edge(Graph.q2id[q1], Graph.q2id[q2], weight=weight)
            Graph.p2weight[(Graph.q2id[q1], Graph.q2id[q2])] = weight
            Graph.p2weight[(Graph.q2id[q2], Graph.q2id[q1])] = weight
            index += 1
        LogUtil.log('INFO', 'len(questions)=%d' % len(Graph.q2id))
        # fout.close()

        # Graph.G = nx.Graph()
        # for line in open('%s/graph_question2id.train.txt' % cf.get('DEFAULT', 'devel_pt')):
        #     head, tail, label, weight = line.strip().split()
        #     head = int(head)
        #     tail = int(tail)
        #     label = int(label)
        #     weight = Graph.p2weight[(head, tail)]
        #     Graph.G.add_edge(head, tail, weight=weight)
        # for line in open('%s/graph_question2id.test.txt' % cf.get('DEFAULT', 'devel_pt')):
        #     head, tail, weight = line.strip().split()
        #     head = int(head)
        #     tail = int(tail)
        #     weight = Graph.p2weight[(head, tail)]
        #     Graph.G.add_edge(head, tail, weight=weight)
        LogUtil.log('INFO', 'Graph constructed.')

    @staticmethod
    def init_graph_nostrip(cf, argv):
        Graph.q2id = {}

        fin = csv.reader(open('%s/train.csv' % cf.get('DEFAULT', 'origin_pt')))
        fin.next()
        fout = open('%s/graph_question2id_nostrip.train.txt' % cf.get('DEFAULT', 'devel_pt'), 'w')
        for p in fin:
            q1 = str(p[3])
            q2 = str(p[4])
            label = p[5]
            if q1 not in Graph.q2id:
                Graph.q2id[q1] = len(Graph.q2id)
            if q2 not in Graph.q2id:
                Graph.q2id[q2] = len(Graph.q2id)
            print >> fout, Graph.q2id[q1], Graph.q2id[q2], label
        LogUtil.log('INFO', 'len(questions)=%d' % len(Graph.q2id))
        fout.close()

        fin = csv.reader(open('%s/test.csv' % cf.get('DEFAULT', 'origin_pt')))
        fin.next()
        fout = open('%s/graph_question2id_nostrip.test.txt' % cf.get('DEFAULT', 'devel_pt'), 'w')
        for p in fin:
            q1 = str(p[1])
            q2 = str(p[2])
            if q1 not in Graph.q2id:
                Graph.q2id[q1] = len(Graph.q2id)
            if q2 not in Graph.q2id:
                Graph.q2id[q2] = len(Graph.q2id)
            print >> fout, Graph.q2id[q1], Graph.q2id[q2]
        LogUtil.log('INFO', 'len(questions)=%d' % len(Graph.q2id))
        fout.close()

        Graph.G = nx.Graph()
        for line in open('%s/graph_question2id_nostrip.train.txt' % cf.get('DEFAULT', 'devel_pt')):
            head, tail, label = [int(x) for x in line.split()]
            Graph.G.add_edge(head, tail, weight=label + 1)
        for line in open('%s/graph_question2id_nostrip.test.txt' % cf.get('DEFAULT', 'devel_pt')):
            head, tail = [int(x) for x in line.split()]
            Graph.G.add_edge(head, tail, weight=0)
        LogUtil.log('INFO', 'Graph constructed.')

    @staticmethod
    def init_pagerank(cf, alpha, max_iter):
        Graph.G = nx.Graph()
        for line in open('%s/graph_question2id.train.txt' % cf.get('DEFAULT', 'devel_pt')):
            head, tail, label = [int(x) for x in line.split()]
            Graph.G.add_edge(head, tail)
        for line in open('%s/graph_question2id.test.txt' % cf.get('DEFAULT', 'devel_pt')):
            head, tail = [int(x) for x in line.split()]
            Graph.G.add_edge(head, tail)
        LogUtil.log('INFO', 'Graph for pagerank constructed.')

        Graph.pr = nx.pagerank(Graph.G, alpha=alpha, max_iter=max_iter)
        LogUtil.log('INFO', 'Graph cal pagerank done')

    @staticmethod
    def init_hits_symm(cf, max_iter):
        Graph.G = nx.Graph()
        for line in open('%s/graph_question2id.train.txt' % cf.get('DEFAULT', 'devel_pt')):
            head, tail, label = [int(x) for x in line.split()]
            Graph.G.add_edge(head, tail)
        for line in open('%s/graph_question2id.test.txt' % cf.get('DEFAULT', 'devel_pt')):
            head, tail = [int(x) for x in line.split()]
            Graph.G.add_edge(head, tail)
        LogUtil.log('INFO', 'Graph for hits constructed.')

        Graph.hits_h, Graph.hits_a = nx.hits(Graph.G, max_iter=max_iter)
        LogUtil.log('INFO', 'Graph cal hits done')


    @staticmethod
    def extract_row_graph_edge_max_clique_size(row, *args):
        n2clique = args[0]
        cliques = args[1]

        q1 = str(row['question1']).strip()
        q2 = str(row['question2']).strip()

        qid1 = Graph.q2id[q1]
        qid2 = Graph.q2id[q2]

        edge_max_clique_size = 0

        for clique_id in n2clique[qid1]:
            if qid2 in cliques[clique_id]:
                edge_max_clique_size = max(edge_max_clique_size, len(cliques[clique_id]))

        return [edge_max_clique_size]

    @staticmethod
    def extract_graph_edge_max_clique_size(cf, argv):
        # 设置参数
        feature_name = 'graph_edge_max_clique_size'

        Graph.init_graph(cf, argv)

        n2clique = {}
        cliques = []
        for clique in nx.find_cliques(Graph.G):
            for n in clique:
                if n not in n2clique:
                    n2clique[n] = []
                n2clique[n].append(len(cliques))
            cliques.append(clique)
        LogUtil.log('INFO', 'len(cliques)=%d' % len(cliques))

        # 加载数据文件
        train_data = pd.read_csv('%s/train.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")
        test_data = pd.read_csv('%s/test.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")

        # 特征存储路径
        feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
        train_feature_fp = '%s/%s.train.smat' % (feature_pt, feature_name)
        test_feature_fp = '%s/%s.test.smat' % (feature_pt, feature_name)

        # 抽取特征：train.csv
        train_features = train_data.apply(Graph.extract_row_graph_edge_max_clique_size, axis=1, raw=True, args=[n2clique, cliques])
        LogUtil.log('INFO', 'extract train features (%s) done' % feature_name)
        Feature.save_dataframe(train_features, train_feature_fp)
        LogUtil.log('INFO', 'save train features (%s) done' % feature_name)

        test_features = test_data.apply(Graph.extract_row_graph_edge_max_clique_size, axis=1, raw=True, args=[n2clique, cliques])
        LogUtil.log('INFO', 'extract test features (%s) done' % feature_name)
        Feature.save_dataframe(test_features, test_feature_fp)
        LogUtil.log('INFO', 'save test features (%s) done' % feature_name)

    @staticmethod
    def extract_row_graph_edge_max_clique_size_nostrip(row, *args):
        n2clique = args[0]
        cliques = args[1]

        q1 = str(row['question1'])
        q2 = str(row['question2'])

        qid1 = Graph.q2id[q1]
        qid2 = Graph.q2id[q2]

        edge_max_clique_size = 0

        for clique_id in n2clique[qid1]:
            if qid2 in cliques[clique_id]:
                edge_max_clique_size = max(edge_max_clique_size, len(cliques[clique_id]))

        return [edge_max_clique_size]

    @staticmethod
    def extract_graph_edge_max_clique_size_nostrip(cf, argv):
        # 设置参数
        feature_name = 'graph_edge_max_clique_size_nostrip'

        Graph.init_graph_nostrip(cf, argv)

        n2clique = {}
        cliques = []
        for clique in nx.find_cliques(Graph.G):
            for n in clique:
                if n not in n2clique:
                    n2clique[n] = []
                n2clique[n].append(len(cliques))
            cliques.append(clique)
        LogUtil.log('INFO', 'len(cliques)=%d' % len(cliques))

        # 加载数据文件
        train_data = pd.read_csv('%s/train.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")
        test_data = pd.read_csv('%s/test.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")

        # 特征存储路径
        feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
        train_feature_fp = '%s/%s.train.smat' % (feature_pt, feature_name)
        test_feature_fp = '%s/%s.test.smat' % (feature_pt, feature_name)

        # 抽取特征：train.csv
        train_features = train_data.apply(Graph.extract_row_graph_edge_max_clique_size_nostrip, axis=1, raw=True,
                                          args=[n2clique, cliques])
        LogUtil.log('INFO', 'extract train features (%s) done' % feature_name)
        Feature.save_dataframe(train_features, train_feature_fp)
        LogUtil.log('INFO', 'save train features (%s) done' % feature_name)

        test_features = test_data.apply(Graph.extract_row_graph_edge_max_clique_size_nostrip, axis=1, raw=True,
                                        args=[n2clique, cliques])
        LogUtil.log('INFO', 'extract test features (%s) done' % feature_name)
        Feature.save_dataframe(test_features, test_feature_fp)
        LogUtil.log('INFO', 'save test features (%s) done' % feature_name)

    @staticmethod
    def show_graph_edge_max_clique_size(cf, argv):
        # 设置参数
        feature_name = 'graph_edge_max_clique_size'

        # 加载数据文件
        train_data = pd.read_csv('%s/train.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")
        test_data = pd.read_csv('%s/test.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")
        # 特征存储路径
        feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
        train_feature_fp = '%s/%s.train.smat' % (feature_pt, feature_name)
        test_feature_fp = '%s/%s.test.smat' % (feature_pt, feature_name)
        # 加载特征文件
        train_features = Feature.load_smat(train_feature_fp).data
        test_features = Feature.load_smat(test_feature_fp).data


        plt.figure(figsize=(15, 5))
        # plt.xlim([3, 40])
        plt.hist([train_features[index] for index in range(len(train_features)) if (train_data['is_duplicate'][index] == 0 and train_features[index] > 2)],bins=range(50), normed=False, label='Not Duplicate')
        plt.hist([train_features[index] for index in range(len(train_features)) if (train_data['is_duplicate'][index] == 1 and train_features[index] > 2)],bins=range(50), normed=False, alpha=0.7, label='Duplicate', histtype='bar')
        plt.legend()
        plt.title('Label distribution over %s' % feature_name, fontsize=15)
        plt.xlabel(feature_name, fontsize=15)
        plt.show()

    @staticmethod
    def extract_row_graph_edge_min_clique_size(row, *args):
        n2clique = args[0]
        cliques = args[1]

        q1 = str(row['question1']).strip()
        q2 = str(row['question2']).strip()

        qid1 = Graph.q2id[q1]
        qid2 = Graph.q2id[q2]

        edge_min_clique_size = 5000000

        for clique_id in n2clique[qid1]:
            if qid2 in cliques[clique_id]:
                edge_min_clique_size = min(edge_min_clique_size, len(cliques[clique_id]))

        return [edge_min_clique_size]

    @staticmethod
    def extract_graph_edge_min_clique_size(cf, argv):
        # 设置参数
        feature_name = 'graph_edge_min_clique_size'

        Graph.init_graph(cf, argv)

        n2clique = {}
        cliques = []
        for clique in nx.find_cliques(Graph.G):
            for n in clique:
                if n not in n2clique:
                    n2clique[n] = []
                n2clique[n].append(len(cliques))
            cliques.append(clique)
        LogUtil.log('INFO', 'len(cliques)=%d' % len(cliques))

        # 加载数据文件
        train_data = pd.read_csv('%s/train.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")
        test_data = pd.read_csv('%s/test.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")

        # 特征存储路径
        feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
        train_feature_fp = '%s/%s.train.smat' % (feature_pt, feature_name)
        test_feature_fp = '%s/%s.test.smat' % (feature_pt, feature_name)

        # 抽取特征：train.csv
        train_features = train_data.apply(Graph.extract_row_graph_edge_min_clique_size, axis=1, raw=True,
                                          args=[n2clique, cliques])
        LogUtil.log('INFO', 'extract train features (%s) done' % feature_name)
        Feature.save_dataframe(train_features, train_feature_fp)
        LogUtil.log('INFO', 'save train features (%s) done' % feature_name)

        test_features = test_data.apply(Graph.extract_row_graph_edge_min_clique_size, axis=1, raw=True,
                                        args=[n2clique, cliques])
        LogUtil.log('INFO', 'extract test features (%s) done' % feature_name)
        Feature.save_dataframe(test_features, test_feature_fp)
        LogUtil.log('INFO', 'save test features (%s) done' % feature_name)

    @staticmethod
    def extract_row_graph_num_clique(row, *args):
        n2clique = args[0]
        cliques = args[1]

        q1 = str(row['question1']).strip()
        q2 = str(row['question2']).strip()

        qid1 = Graph.q2id[q1]
        qid2 = Graph.q2id[q2]

        num_clique = 0

        for clique_id in n2clique[qid1]:
            if qid2 in cliques[clique_id]:
                num_clique += 1

        return [num_clique]

    @staticmethod
    def extract_graph_num_clique(cf, argv):
        """
        计算包含边的团的数量
        :param cf:
        :param argv:
        :return:
        """
        # 设置参数
        feature_name = 'graph_num_clique'

        Graph.init_graph(cf, argv)

        n2clique = {}  # 节点到团的映射
        cliques = []  # 所有团
        for clique in nx.find_cliques(Graph.G):
            for n in clique:
                if n not in n2clique:
                    n2clique[n] = []
                n2clique[n].append(len(cliques))
            cliques.append(clique)
        LogUtil.log('INFO', 'len(cliques)=%d' % len(cliques))

        # 加载数据文件
        train_data = pd.read_csv('%s/train.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")
        test_data = pd.read_csv('%s/test.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")

        # 特征存储路径
        feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
        train_feature_fp = '%s/%s.train.smat' % (feature_pt, feature_name)
        test_feature_fp = '%s/%s.test.smat' % (feature_pt, feature_name)

        # 抽取特征：train.csv
        train_features = train_data.apply(Graph.extract_row_graph_num_clique, axis=1, raw=True,
                                          args=[n2clique, cliques])
        LogUtil.log('INFO', 'extract train features (%s) done' % feature_name)
        Feature.save_dataframe(train_features, train_feature_fp)
        LogUtil.log('INFO', 'save train features (%s) done' % feature_name)

        test_features = test_data.apply(Graph.extract_row_graph_num_clique, axis=1, raw=True,
                                        args=[n2clique, cliques])
        LogUtil.log('INFO', 'extract test features (%s) done' % feature_name)
        Feature.save_dataframe(test_features, test_feature_fp)
        LogUtil.log('INFO', 'save test features (%s) done' % feature_name)

    @staticmethod
    def extract_row_graph_edge_cc_size(row, *args):
        n2cc = args[0]
        ccs = args[1]

        q1 = str(row['question1']).strip()
        q2 = str(row['question2']).strip()

        qid1 = Graph.q2id[q1]
        qid2 = Graph.q2id[q2]

        edge_cc_size = len(ccs[n2cc[qid1]])

        return [edge_cc_size]

    @staticmethod
    def extract_graph_edge_cc_size(cf, argv):
        # 设置参数
        feature_name = 'graph_edge_cc_size'

        Graph.init_graph(cf, argv)

        n2cc = {}
        ccs = []
        for cc in nx.connected_components(Graph.G):
            for n in cc:
                if n in n2cc:
                    LogUtil.log('WARNING', '%d already in n2cc(=%d)' % (n, n2cc[n]))
                n2cc[n] = len(ccs)
            ccs.append(cc)
        LogUtil.log('INFO', 'len(cliques)=%d' % len(ccs))

        # 加载数据文件
        train_data = pd.read_csv('%s/train.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")
        test_data = pd.read_csv('%s/test.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")

        # 特征存储路径
        feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
        train_feature_fp = '%s/%s.train.smat' % (feature_pt, feature_name)
        test_feature_fp = '%s/%s.test.smat' % (feature_pt, feature_name)

        # 抽取特征：train.csv
        train_features = train_data.apply(Graph.extract_row_graph_edge_cc_size, axis=1, raw=True,
                                          args=[n2cc, ccs])
        LogUtil.log('INFO', 'extract train features (%s) done' % feature_name)
        Feature.save_dataframe(train_features, train_feature_fp)
        LogUtil.log('INFO', 'save train features (%s) done' % feature_name)

        test_features = test_data.apply(Graph.extract_row_graph_edge_cc_size, axis=1, raw=True,
                                        args=[n2cc, ccs])
        LogUtil.log('INFO', 'extract test features (%s) done' % feature_name)
        Feature.save_dataframe(test_features, test_feature_fp)
        LogUtil.log('INFO', 'save test features (%s) done' % feature_name)

    @staticmethod
    def extract_row_graph_node_max_clique_size(row, *args):
        n2clique = args[0]
        cliques = args[1]

        q1 = str(row['question1']).strip()
        q2 = str(row['question2']).strip()

        qid1 = Graph.q2id[q1]
        qid2 = Graph.q2id[q2]

        lnode_max_clique_size = 0
        rnode_max_clique_size = 0

        for clique_id in n2clique[qid1]:
            lnode_max_clique_size = max(lnode_max_clique_size, len(cliques[clique_id]))

        for clique_id in n2clique[qid2]:
            rnode_max_clique_size = max(rnode_max_clique_size, len(cliques[clique_id]))

        return [lnode_max_clique_size, rnode_max_clique_size, max(lnode_max_clique_size, rnode_max_clique_size), min(lnode_max_clique_size, rnode_max_clique_size)]


    @staticmethod
    def extract_graph_node_max_clique_size(cf, argv):
        # 设置参数
        feature_name = 'graph_node_max_clique_size'

        Graph.init_graph(cf, argv)

        n2clique = {}
        cliques = []
        for clique in nx.find_cliques(Graph.G):
            for n in clique:
                if n not in n2clique:
                    n2clique[n] = []
                n2clique[n].append(len(cliques))
            cliques.append(clique)
        LogUtil.log('INFO', 'len(cliques)=%d' % len(cliques))

        # 加载数据文件
        train_data = pd.read_csv('%s/train.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")
        test_data = pd.read_csv('%s/test.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")

        # 特征存储路径
        feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
        train_feature_fp = '%s/%s.train.smat' % (feature_pt, feature_name)
        test_feature_fp = '%s/%s.test.smat' % (feature_pt, feature_name)

        # 抽取特征：train.csv
        train_features = train_data.apply(Graph.extract_row_graph_node_max_clique_size, axis=1, raw=True,
                                          args=[n2clique, cliques])
        LogUtil.log('INFO', 'extract train features (%s) done' % feature_name)
        Feature.save_dataframe(train_features, train_feature_fp)
        LogUtil.log('INFO', 'save train features (%s) done' % feature_name)

        test_features = test_data.apply(Graph.extract_row_graph_node_max_clique_size, axis=1, raw=True,
                                        args=[n2clique, cliques])
        LogUtil.log('INFO', 'extract test features (%s) done' % feature_name)
        Feature.save_dataframe(test_features, test_feature_fp)
        LogUtil.log('INFO', 'save test features (%s) done' % feature_name)

    @staticmethod
    def extract_row_graph_pagerank_symm(row):
        q1 = str(row['question1']).strip()
        q2 = str(row['question2']).strip()

        qid1 = Graph.q2id[q1]
        qid2 = Graph.q2id[q2]

        pr1 = Graph.pr[qid1] * 1e6
        pr2 = Graph.pr[qid2] * 1e6

        return [pr1, pr2, max(pr1, pr2), min(pr1, pr2), (pr1 + pr2) / 2.]

    @staticmethod
    def extract_graph_pagerank_symm(cf, argv):
        # 设置参数
        alpha = float(argv[0])
        max_iter = int(argv[1])
        feature_name = 'graph_pagerank_symm_%.2f_%d' % (alpha, max_iter)

        Graph.init_graph(cf, argv)

        Graph.init_pagerank(cf, alpha, max_iter)

        # 加载数据文件
        train_data = pd.read_csv('%s/train.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")
        test_data = pd.read_csv('%s/test.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")

        # 特征存储路径
        feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
        train_feature_fp = '%s/%s.train.smat' % (feature_pt, feature_name)
        test_feature_fp = '%s/%s.test.smat' % (feature_pt, feature_name)

        # 抽取特征：train.csv
        train_features = train_data.apply(Graph.extract_row_graph_pagerank_symm, axis=1, raw=True)
        LogUtil.log('INFO', 'extract train features (%s) done' % feature_name)
        Feature.save_dataframe(train_features, train_feature_fp)
        LogUtil.log('INFO', 'save train features (%s) done' % feature_name)

        test_features = test_data.apply(Graph.extract_row_graph_pagerank_symm, axis=1, raw=True)
        LogUtil.log('INFO', 'extract test features (%s) done' % feature_name)
        Feature.save_dataframe(test_features, test_feature_fp)
        LogUtil.log('INFO', 'save test features (%s) done' % feature_name)

    @staticmethod
    def extract_row_graph_hits_symm(row):
        q1 = str(row['question1']).strip()
        q2 = str(row['question2']).strip()

        qid1 = Graph.q2id[q1]
        qid2 = Graph.q2id[q2]

        h1 = Graph.hits_h[qid1] * 1e6
        h2 = Graph.hits_h[qid2] * 1e6

        a1 = Graph.hits_a[qid1] * 1e6
        a2 = Graph.hits_a[qid2] * 1e6

        return [h1, h2, a1, a2,
                max(h1, h2), max(a1, a2),
                min(h1, h2), min(a1, a2),
                (h1 + h2) / 2., (a1 + a2) / 2.]

    @staticmethod
    def extract_graph_hits_symm(cf, argv):
        # 设置参数
        max_iter = int(argv[0])
        feature_name = 'graph_hits_symm_%d' % max_iter

        Graph.init_graph(cf, argv)

        Graph.init_hits_symm(cf, max_iter)

        # 加载数据文件
        train_data = pd.read_csv('%s/train.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")
        test_data = pd.read_csv('%s/test.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")

        # 特征存储路径
        feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
        train_feature_fp = '%s/%s.train.smat' % (feature_pt, feature_name)
        test_feature_fp = '%s/%s.test.smat' % (feature_pt, feature_name)

        # 抽取特征：train.csv
        train_features = train_data.apply(Graph.extract_row_graph_hits_symm, axis=1, raw=True)
        LogUtil.log('INFO', 'extract train features (%s) done' % feature_name)
        Feature.save_dataframe(train_features, train_feature_fp)
        LogUtil.log('INFO', 'save train features (%s) done' % feature_name)

        test_features = test_data.apply(Graph.extract_row_graph_hits_symm, axis=1, raw=True)
        LogUtil.log('INFO', 'extract test features (%s) done' % feature_name)
        Feature.save_dataframe(test_features, test_feature_fp)
        LogUtil.log('INFO', 'save test features (%s) done' % feature_name)

    @staticmethod
    def extract_graph_mc_cc_rate(cf, argv):
        # 设置参数
        feature_name = 'graph_mc_cc_rate'

        # 加载数据文件
        train_mc_fs = Feature.load(
            '%s/%s.train.smat' % (cf.get('DEFAULT', 'feature_question_pair_pt'), 'graph_edge_max_clique_size')).toarray()
        train_cc_fs = Feature.load(
            '%s/%s.train.smat' % (cf.get('DEFAULT', 'feature_question_pair_pt'), 'graph_edge_cc_size')).toarray()

        train_features = [[1. * train_mc_fs[index][0] / train_cc_fs[index][0]] for index in range(len(train_mc_fs))]
        train_features = sparse.csr_matrix(train_features)

        # 加载数据文件
        test_mc_fs = Feature.load(
            '%s/%s.test.smat' % (
            cf.get('DEFAULT', 'feature_question_pair_pt'), 'graph_edge_max_clique_size')).toarray()
        test_cc_fs = Feature.load(
            '%s/%s.test.smat' % (cf.get('DEFAULT', 'feature_question_pair_pt'), 'graph_edge_cc_size')).toarray()

        test_features = [[1. * test_mc_fs[index][0] / test_cc_fs[index][0]] for index in range(len(test_mc_fs))]
        test_features = sparse.csr_matrix(test_features)

        # 特征存储路径
        feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
        train_feature_fp = '%s/%s.train.smat' % (feature_pt, feature_name)
        test_feature_fp = '%s/%s.test.smat' % (feature_pt, feature_name)

        # 抽取特征：train.csv
        Feature.save_smat(train_features, train_feature_fp)
        LogUtil.log('INFO', 'save train features (%s) done' % feature_name)

        Feature.save_smat(test_features, test_feature_fp)
        LogUtil.log('INFO', 'save test features (%s) done' % feature_name)

    @staticmethod
    def extract_row_graph_shortest_path(row):
        q1 = str(row['question1']).strip()
        q2 = str(row['question2']).strip()

        qid1 = Graph.q2id[q1]
        qid2 = Graph.q2id[q2]

        shortest_path = -1
        Graph.G.remove_edge(qid1, qid2)
        if nx.has_path(Graph.G, qid1, qid2):
            shortest_path = nx.dijkstra_path_length(Graph.G, qid1, qid2)
        Graph.G.add_edge(qid1, qid2, weight=Graph.p2weight[(qid1, qid2)])

        return [shortest_path]

    @staticmethod
    def extract_graph_shortest_path(cf, argv):
        # 路径权重特征名
        weight_feature_name = argv[0]  # e.g. my_tfidf_word_match_share
        # 抽取特征的数据集名称
        dataset_name = argv[1]  # e.g. train
        # 划分 part 数目
        part_num = int(argv[2])
        # part 的 ID
        part_id = int(argv[3])
        # reverse
        reverse = argv[4]
        # 特征第几维
        weight_feature_id = int(argv[5])
        # 设置参数
        feature_name = 'graph_shortest_path_%s_%s' % (weight_feature_name, reverse)

        Graph.init_graph_with_weight(cf, weight_feature_name, weight_feature_id, reverse)

        # 加载数据文件
        data = pd.read_csv('%s/%s.csv' % (cf.get('DEFAULT', 'origin_pt'), dataset_name)).fillna(value="")
        begin_id = int(1. * len(data) / part_num * part_id)
        end_id = int(1. * len(data) / part_num * (part_id + 1))

        # 存储路径
        feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
        data_feature_fp = '%s/%s.%s.smat.%03d' % (feature_pt, feature_name, dataset_name, part_id)

        # 抽取特征
        features = data[begin_id:end_id].apply(Graph.extract_row_graph_shortest_path, axis=1, raw=True)
        Feature.save_dataframe(features, data_feature_fp)
        LogUtil.log('INFO', 'save train features (%s, %s, %d, %d) done' % (feature_name, dataset_name, part_num, part_id))

    @staticmethod
    def merge_graph_shortest_path(cf, argv):
        # 路径权重特征名
        weight_feature_name = argv[0]  # e.g. my_tfidf_word_match_share
        # 抽取特征的数据集名称
        dataset_name = argv[1]  # e.g. train
        # 划分 part 数目
        part_num = int(argv[2])
        # reverse
        reverse = argv[3]
        feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
        # 设置参数
        feature_name = 'graph_shortest_path_%s_%s' % (weight_feature_name, reverse)

        features = None
        for part_id in range(part_num):
            features_part_fp = '%s/%s.%s.smat.%03d' % (feature_pt, feature_name, dataset_name, part_id)
            features_part = Feature.load(features_part_fp)
            if features is None:
                features = features_part
            else:
                features = Feature.merge_row(features, features_part)

        features_fp = '%s/%s.%s.smat' % (feature_pt, feature_name, dataset_name)
        Feature.save_smat(features, features_fp)
        LogUtil.log('INFO',
                    'save train features (%s, %s, %d) done' % (feature_name, dataset_name, part_num))

    @staticmethod
    def extract_row_clique_size_e3_other_edge(row, *args):
        n2clique = args[0]
        cliques = args[1]

        q1 = str(row['question1']).strip()
        q2 = str(row['question2']).strip()

        qid1 = Graph.q2id[q1]
        qid2 = Graph.q2id[q2]

        edge_max_clique_size = 0

        for clique_id in n2clique[qid1]:
            if qid2 in cliques[clique_id]:
                edge_max_clique_size = max(edge_max_clique_size, len(cliques[clique_id]))

        aggregation_mode = ["mean", "std", "max", "min"]
        aggregation_mode, aggregator = Distance._check_aggregation_mode(aggregation_mode)

        sub = []
        add = []
        fs = [0]
        if edge_max_clique_size == 3:
            for clique_id in n2clique[qid1]:
                if (3 == len(cliques[clique_id])) and (qid2 in cliques[clique_id]):
                    qid3 = sum(cliques[clique_id]) - qid1 - qid2
                    w1 = Graph.p2weight[(qid1, qid3)]
                    w2 = Graph.p2weight[(qid2, qid3)]
                    sub.append(abs(w1 - w2))
                    add.append(w1 + w2)
                    fs[0] += 1.
        else:
            sub.append(-1)
            add.append(-1)

        for agg in aggregator:
            try:
                s = agg(sub)
            except:
                s = config.MISSING_VALUE_NUMERIC
            fs.append(s)
        for agg in aggregator:
            try:
                s = agg(add)
            except:
                s = config.MISSING_VALUE_NUMERIC
            fs.append(s)

        # 计数器
        Graph.counter += 1
        if Graph.counter % 1000 == 0:
            LogUtil.log('INFO', 'Graph.counter=%d' % Graph.counter)

        return fs

    @staticmethod
    def extractor_clique_size_e3_other_edge(cf, argv):
        # 路径权重特征名
        weight_feature_name = argv[0]  # e.g. my_tfidf_word_match_share
        # 抽取特征的数据集名称
        dataset_name = argv[1]  # e.g. train
        # 划分 part 数目
        part_num = int(argv[2])
        # part 的 ID
        part_id = int(argv[3])
        # 特征第几维
        weight_feature_id = int(argv[4])
        # 设置参数
        feature_name = 'graph_clique_size_e3_other_edge_%s' % weight_feature_name

        Graph.init_graph_with_weight(cf, weight_feature_name, weight_feature_id)

        n2clique = {}
        cliques = []
        for clique in nx.find_cliques(Graph.G):
            for n in clique:
                if n not in n2clique:
                    n2clique[n] = []
                n2clique[n].append(len(cliques))
            cliques.append(clique)
        LogUtil.log('INFO', 'len(cliques)=%d' % len(cliques))

        # 加载数据文件
        data = pd.read_csv('%s/%s.csv' % (cf.get('DEFAULT', 'origin_pt'), dataset_name)).fillna(value="")
        begin_id = int(1. * len(data) / part_num * part_id)
        end_id = int(1. * len(data) / part_num * (part_id + 1))

        # 存储路径
        feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
        if 1 == part_num:
            data_feature_fp = '%s/%s.%s.smat' % (feature_pt, feature_name, dataset_name)
        else:
            data_feature_fp = '%s/%s.%s.smat.%03d' % (feature_pt, feature_name, dataset_name, part_id)

        # 抽取特征
        features = data[begin_id:end_id].apply(Graph.extract_row_clique_size_e3_other_edge, axis=1, raw=True, args=[n2clique, cliques])
        Feature.save_dataframe(features, data_feature_fp)
        LogUtil.log('INFO',
                    'save train features (%s, %s, %d, %d) done' % (feature_name, dataset_name, part_num, part_id))


    @staticmethod
    def run(cf, argv):
        cmd = argv[0]

        if 'extract_graph_edge_max_clique_size' == cmd:
            Graph.extract_graph_edge_max_clique_size(cf, argv[1:])
        elif 'extract_graph_edge_max_clique_size_nostrip' == cmd:
                Graph.extract_graph_edge_max_clique_size_nostrip(cf, argv[1:])
        elif 'extract_graph_num_clique' == cmd:
            Graph.extract_graph_num_clique(cf, argv[1:])
        elif 'extract_graph_edge_min_clique_size' == cmd:
            Graph.extract_graph_edge_min_clique_size(cf, argv[1:])
        elif 'extract_graph_edge_cc_size' == cmd:
            Graph.extract_graph_edge_cc_size(cf, argv[1:])
        elif 'extract_graph_node_max_clique_size' == cmd:
            Graph.extract_graph_node_max_clique_size(cf, argv[1:])
        elif 'extract_graph_pagerank_symm' == cmd:
            Graph.extract_graph_pagerank_symm(cf, argv[1:])
        elif 'extract_graph_hits_symm' == cmd:
            Graph.extract_graph_hits_symm(cf, argv[1:])
        elif 'show_graph_edge_max_clique_size' == cmd:
            Graph.show_graph_edge_max_clique_size(cf, argv[1:])
        elif 'extract_graph_mc_cc_rate' == cmd:
            Graph.extract_graph_mc_cc_rate(cf, argv[1:])
        elif 'extract_graph_shortest_path' == cmd:
            Graph.extract_graph_shortest_path(cf, argv[1:])
        elif 'merge_graph_shortest_path' == cmd:
            Graph.merge_graph_shortest_path(cf, argv[1:])
        elif 'extractor_clique_size_e3_other_edge' == cmd:
            Graph.extractor_clique_size_e3_other_edge(cf, argv[1:])
        else:
            LogUtil.log('WARNING', 'NO CMD')


class Count(object):

    def __init__(self):
        pass

    @staticmethod
    def extract_row_eng_char_count(row):
        s = 'abcdefghijklmnopqrstuvwxyz'

        q1 = str(row['question1']).strip().lower()
        q2 = str(row['question2']).strip().lower()

        fs1 = [0] * 26
        fs2 = [0] * 26

        for index in range(len(q1)):
            c = q1[index]
            if 0 <= s.find(c):
                fs1[s.find(c)] += 1
        for index in range(len(q2)):
            c = q2[index]
            if 0 <= s.find(c):
                fs2[s.find(c)] += 1

        return fs1 + fs2 + list(abs(np.array(fs1) - np.array(fs2)))

    @staticmethod
    def extract_eng_char_count(cf, argv):
        # 设置参数
        feature_name = 'eng_char_count'

        # 加载数据文件
        train_data = pd.read_csv('%s/train.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")
        test_data = pd.read_csv('%s/test.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")

        # 特征存储路径
        feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
        train_feature_fp = '%s/%s.train.smat' % (feature_pt, feature_name)
        test_feature_fp = '%s/%s.test.smat' % (feature_pt, feature_name)

        # 抽取特征：train.csv
        train_features = train_data.apply(Count.extract_row_eng_char_count, axis=1, raw=True)
        LogUtil.log('INFO', 'extract train features (%s) done' % feature_name)
        Feature.save_dataframe(train_features, train_feature_fp)
        LogUtil.log('INFO', 'save train features (%s) done' % feature_name)

        test_features = test_data.apply(Count.extract_row_eng_char_count, axis=1, raw=True)
        LogUtil.log('INFO', 'extract test features (%s) done' % feature_name)
        Feature.save_dataframe(test_features, test_feature_fp)
        LogUtil.log('INFO', 'save test features (%s) done' % feature_name)

    @staticmethod
    def run(cf, argv):
        cmd = argv[0]

        if 'extract_eng_char_count' == cmd:
            Count.extract_eng_char_count(cf, argv[1:])
        else:
            LogUtil.log('WARNING', 'NO CMD')


class Distance(object):
    snowball_stemmer = SnowballStemmer('english')

    def __init__(self):
        pass

    @staticmethod
    def extract_row_jaccard_coef_ngram(row):
        q1_words = [Distance.snowball_stemmer.stem(word).encode('utf-8') for word in
                    nltk.word_tokenize(Preprocessor.clean_text(str(row['question1']).decode('utf-8')))]
        q2_words = [Distance.snowball_stemmer.stem(word).encode('utf-8') for word in
                    nltk.word_tokenize(Preprocessor.clean_text(str(row['question2']).decode('utf-8')))]

        fs = []
        for n in range(1, 4):
            q1_ngrams = ngram_utils._ngrams(q1_words, n)
            q2_ngrams = ngram_utils._ngrams(q2_words, n)
            fs.append(dist_utils._jaccard_coef(q1_ngrams, q2_ngrams))

        return fs


    @staticmethod
    def extract_jaccard_coef_ngram(cf, argv):
        # 设置参数
        feature_name = 'jaccard_coef_ngram'

        # 加载数据文件
        train_data = pd.read_csv('%s/train.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")
        test_data = pd.read_csv('%s/test.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")

        # 特征存储路径
        feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
        train_feature_fp = '%s/%s.train.smat' % (feature_pt, feature_name)
        test_feature_fp = '%s/%s.test.smat' % (feature_pt, feature_name)

        # 抽取特征：train.csv
        train_features = train_data.apply(Distance.extract_row_jaccard_coef_ngram, axis=1, raw=True)
        LogUtil.log('INFO', 'extract train features (%s) done' % feature_name)
        Feature.save_dataframe(train_features, train_feature_fp)
        LogUtil.log('INFO', 'save train features (%s) done' % feature_name)

        test_features = test_data.apply(Distance.extract_row_jaccard_coef_ngram, axis=1, raw=True)
        LogUtil.log('INFO', 'extract test features (%s) done' % feature_name)
        Feature.save_dataframe(test_features, test_feature_fp)
        LogUtil.log('INFO', 'save test features (%s) done' % feature_name)

    @staticmethod
    def extract_row_dice_dis_ngram(row):
        q1_words = [Distance.snowball_stemmer.stem(word).encode('utf-8') for word in
                    nltk.word_tokenize(Preprocessor.clean_text(str(row['question1']).decode('utf-8')))]
        q2_words = [Distance.snowball_stemmer.stem(word).encode('utf-8') for word in
                    nltk.word_tokenize(Preprocessor.clean_text(str(row['question2']).decode('utf-8')))]

        fs = []
        for n in range(1, 4):
            q1_ngrams = ngram_utils._ngrams(q1_words, n)
            q2_ngrams = ngram_utils._ngrams(q2_words, n)
            fs.append(dist_utils._dice_dist(q1_ngrams, q2_ngrams))

        return fs

    @staticmethod
    def extract_dice_dis_ngram(cf, argv):
        # 设置参数
        feature_name = 'dice_dis_ngram'

        # 加载数据文件
        train_data = pd.read_csv('%s/train.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")
        test_data = pd.read_csv('%s/test.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")

        # 特征存储路径
        feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
        train_feature_fp = '%s/%s.train.smat' % (feature_pt, feature_name)
        test_feature_fp = '%s/%s.test.smat' % (feature_pt, feature_name)

        # 抽取特征：train.csv
        train_features = train_data.apply(Distance.extract_row_dice_dis_ngram, axis=1, raw=True)
        LogUtil.log('INFO', 'extract train features (%s) done' % feature_name)
        Feature.save_dataframe(train_features, train_feature_fp)
        LogUtil.log('INFO', 'save train features (%s) done' % feature_name)

        test_features = test_data.apply(Distance.extract_row_dice_dis_ngram, axis=1, raw=True)
        LogUtil.log('INFO', 'extract test features (%s) done' % feature_name)
        Feature.save_dataframe(test_features, test_feature_fp)
        LogUtil.log('INFO', 'save test features (%s) done' % feature_name)

    @staticmethod
    def extract_row_edit_dis(row):
        q1 = str(row['question1']).strip()
        q2 = str(row['question2']).strip()
        q1_stem = ' '.join([Distance.snowball_stemmer.stem(word).encode('utf-8') for word in
                    nltk.word_tokenize(Preprocessor.clean_text(str(row['question1']).decode('utf-8')))])
        q2_stem = ' '.join([Distance.snowball_stemmer.stem(word).encode('utf-8') for word in
                    nltk.word_tokenize(Preprocessor.clean_text(str(row['question2']).decode('utf-8')))])
        return [dist_utils._edit_dist(q1, q2), dist_utils._edit_dist(q1_stem, q2_stem)]

    @staticmethod
    def extract_edit_dis(cf, argv):
        # 设置参数
        feature_name = 'edit_dis'

        # 加载数据文件
        train_data = pd.read_csv('%s/train.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")
        test_data = pd.read_csv('%s/test.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")

        # 特征存储路径
        feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
        train_feature_fp = '%s/%s.train.smat' % (feature_pt, feature_name)
        test_feature_fp = '%s/%s.test.smat' % (feature_pt, feature_name)

        # 抽取特征：train.csv
        train_features = train_data.apply(Distance.extract_row_edit_dis, axis=1, raw=True)
        LogUtil.log('INFO', 'extract train features (%s) done' % feature_name)
        Feature.save_dataframe(train_features, train_feature_fp)
        LogUtil.log('INFO', 'save train features (%s) done' % feature_name)

        train_label = train_data['is_duplicate'].values[:]
        train_features = train_features.tolist()
        train_features = np.array(train_features)

        for i in range(len(train_features[0])):
            corr = np_utils._corr(train_features[:, i], train_label)
            LogUtil.log('INFO', 'corr(%s_%d)=%f' % (feature_name, i, corr))

        test_features = test_data.apply(Distance.extract_row_edit_dis, axis=1, raw=True)
        LogUtil.log('INFO', 'extract test features (%s) done' % feature_name)
        Feature.save_dataframe(test_features, test_feature_fp)
        LogUtil.log('INFO', 'save test features (%s) done' % feature_name)

    @staticmethod
    def extract_row_edit_dis_ngram(row):
        q1_words = [Distance.snowball_stemmer.stem(word).encode('utf-8') for word in
                    nltk.word_tokenize(Preprocessor.clean_text(str(row['question1']).decode('utf-8')))]
        q2_words = [Distance.snowball_stemmer.stem(word).encode('utf-8') for word in
                    nltk.word_tokenize(Preprocessor.clean_text(str(row['question2']).decode('utf-8')))]

        fs = []
        aggregation_mode_prev = ["mean", "max", "min", "median"]
        aggregation_mode = ["mean", "std", "max", "min", "median"]

        aggregation_mode, aggregator = Distance._check_aggregation_mode(aggregation_mode)
        aggregation_mode_prev, aggregator_prev = Distance._check_aggregation_mode(aggregation_mode_prev)

        for n_ngram in range(1, 4):
            q1_ngrams = ngram_utils._ngrams(q1_words, n_ngram)
            q2_ngrams = ngram_utils._ngrams(q2_words, n_ngram)

            val_list = []
            for w1 in q1_ngrams:
                _val_list = []
                for w2 in q2_ngrams:
                    s = dist_utils._edit_dist(w1, w2)
                    _val_list.append(s)
                if len(_val_list) == 0:
                    _val_list = [config.MISSING_VALUE_NUMERIC]
                val_list.append(_val_list)
            if len(val_list) == 0:
                val_list = [[config.MISSING_VALUE_NUMERIC]]

            res = np.zeros(len(aggregator_prev) * len(aggregator), dtype=float)
            for m, aggregator_prev_i in enumerate(aggregator_prev):
                for n, aggregator_i in enumerate(aggregator):
                    idx = m * len(aggregator) + n
                    try:
                        tmp = []
                        for l in val_list:
                            try:
                                s = aggregator_prev_i(l)
                            except:
                                s = config.MISSING_VALUE_NUMERIC
                            tmp.append(s)
                    except:
                        tmp = [config.MISSING_VALUE_NUMERIC]
                    try:
                        s = aggregator_i(tmp)
                    except:
                        s = config.MISSING_VALUE_NUMERIC
                    res[idx] = s

            fs.extend(res)
        return fs

    @staticmethod
    def _check_aggregation_mode(aggregation_mode):
        valid_aggregation_modes = ["", "size", "mean", "std", "max", "min", "median"]
        if isinstance(aggregation_mode, str):
            assert aggregation_mode.lower() in valid_aggregation_modes, "Wrong aggregation_mode: %s"%aggregation_mode
            aggregation_mode = [aggregation_mode.lower()]
        elif isinstance(aggregation_mode, list):
            for m in aggregation_mode:
                assert m.lower() in valid_aggregation_modes, "Wrong aggregation_mode: %s"%m
            aggregation_mode = [m.lower() for m in aggregation_mode]

        aggregator = [None if m == "" else getattr(np, m) for m in aggregation_mode]

        return aggregation_mode, aggregator

    @staticmethod
    def extract_edit_dis_ngram(cf, argv):
        # 设置参数
        feature_name = 'edit_dis_ngram'

        # 加载数据文件
        train_data = pd.read_csv('%s/train.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")
        test_data = pd.read_csv('%s/test.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")

        # 特征存储路径
        feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
        train_feature_fp = '%s/%s.train.smat' % (feature_pt, feature_name)
        test_feature_fp = '%s/%s.test.smat' % (feature_pt, feature_name)

        # 抽取特征：train.csv
        train_features = train_data.apply(Distance.extract_row_edit_dis_ngram, axis=1, raw=True)
        LogUtil.log('INFO', 'extract train features (%s) done' % feature_name)
        Feature.save_dataframe(train_features, train_feature_fp)
        LogUtil.log('INFO', 'save train features (%s) done' % feature_name)

        test_features = test_data.apply(Distance.extract_row_edit_dis_ngram, axis=1, raw=True)
        LogUtil.log('INFO', 'extract test features (%s) done' % feature_name)
        Feature.save_dataframe(test_features, test_feature_fp)
        LogUtil.log('INFO', 'save test features (%s) done' % feature_name)

    @staticmethod
    def extract_row_compression_dis(row):
        q1 = str(row['question1']).strip()
        q2 = str(row['question2']).strip()
        q1_stem = ' '.join([Distance.snowball_stemmer.stem(word).encode('utf-8') for word in
                            nltk.word_tokenize(Preprocessor.clean_text(str(row['question1']).decode('utf-8')))])
        q2_stem = ' '.join([Distance.snowball_stemmer.stem(word).encode('utf-8') for word in
                            nltk.word_tokenize(Preprocessor.clean_text(str(row['question2']).decode('utf-8')))])

        return [dist_utils._compression_dist(q1, q2), dist_utils._compression_dist(q1_stem, q2_stem)]

    @staticmethod
    def extract_compression_dis(cf, argv):
        # 设置参数
        feature_name = 'cmpression_dis'

        # 加载数据文件
        train_data = pd.read_csv('%s/train.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")
        test_data = pd.read_csv('%s/test.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")

        # 特征存储路径
        feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
        train_feature_fp = '%s/%s.train.smat' % (feature_pt, feature_name)
        test_feature_fp = '%s/%s.test.smat' % (feature_pt, feature_name)

        # 抽取特征：train.csv
        train_features = train_data.apply(Distance.extract_row_compression_dis, axis=1, raw=True)
        LogUtil.log('INFO', 'extract train features (%s) done' % feature_name)
        Feature.save_dataframe(train_features, train_feature_fp)
        LogUtil.log('INFO', 'save train features (%s) done' % feature_name)

        train_label = train_data['is_duplicate'].values[:]
        train_features = train_features.tolist()
        train_features = np.array(train_features)

        for i in range(len(train_features[0])):
            corr = np_utils._corr(train_features[:, i], train_label)
            LogUtil.log('INFO', 'corr(%s_%d)=%f' % (feature_name, i, corr))

        test_features = test_data.apply(Distance.extract_row_compression_dis, axis=1, raw=True)
        LogUtil.log('INFO', 'extract test features (%s) done' % feature_name)
        Feature.save_dataframe(test_features, test_feature_fp)
        LogUtil.log('INFO', 'save test features (%s) done' % feature_name)

    @staticmethod
    def extract_row_compression_dis_ngram(row):
        q1_words = [Distance.snowball_stemmer.stem(word).encode('utf-8') for word in
                    nltk.word_tokenize(Preprocessor.clean_text(str(row['question1']).decode('utf-8')))]
        q2_words = [Distance.snowball_stemmer.stem(word).encode('utf-8') for word in
                    nltk.word_tokenize(Preprocessor.clean_text(str(row['question2']).decode('utf-8')))]

        fs = []
        aggregation_mode_prev = ["mean", "max", "min", "median"]
        aggregation_mode = ["mean", "std", "max", "min", "median"]

        aggregation_mode, aggregator = Distance._check_aggregation_mode(aggregation_mode)
        aggregation_mode_prev, aggregator_prev = Distance._check_aggregation_mode(aggregation_mode_prev)

        for n_ngram in range(1, 4):
            q1_ngrams = ngram_utils._ngrams(q1_words, n_ngram)
            q2_ngrams = ngram_utils._ngrams(q2_words, n_ngram)

            val_list = []
            for w1 in q1_ngrams:
                _val_list = []
                for w2 in q2_ngrams:
                    s = dist_utils._compression_dist(w1, w2)
                    _val_list.append(s)
                if len(_val_list) == 0:
                    _val_list = [config.MISSING_VALUE_NUMERIC]
                val_list.append(_val_list)
            if len(val_list) == 0:
                val_list = [[config.MISSING_VALUE_NUMERIC]]

            res = np.zeros(len(aggregator_prev) * len(aggregator), dtype=float)
            for m, aggregator_prev_i in enumerate(aggregator_prev):
                for n, aggregator_i in enumerate(aggregator):
                    idx = m * len(aggregator) + n
                    try:
                        tmp = []
                        for l in val_list:
                            try:
                                s = aggregator_prev_i(l)
                            except:
                                s = config.MISSING_VALUE_NUMERIC
                            tmp.append(s)
                    except:
                        tmp = [config.MISSING_VALUE_NUMERIC]
                    try:
                        s = aggregator_i(tmp)
                    except:
                        s = config.MISSING_VALUE_NUMERIC
                    res[idx] = s

            fs.extend(res)
        return fs

    @staticmethod
    def extract_compression_dis_ngram(cf, argv):
        # 设置参数
        feature_name = 'compression_dis_ngram'

        # 加载数据文件
        train_data = pd.read_csv('%s/train.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")
        test_data = pd.read_csv('%s/test.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")

        # 特征存储路径
        feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
        train_feature_fp = '%s/%s.train.smat' % (feature_pt, feature_name)
        test_feature_fp = '%s/%s.test.smat' % (feature_pt, feature_name)

        # 抽取特征：train.csv
        train_features = train_data.apply(Distance.extract_row_compression_dis_ngram, axis=1, raw=True)
        LogUtil.log('INFO', 'extract train features (%s) done' % feature_name)
        Feature.save_dataframe(train_features, train_feature_fp)
        LogUtil.log('INFO', 'save train features (%s) done' % feature_name)

        train_label = train_data['is_duplicate'].values[:]
        train_features = train_features.tolist()
        train_features = np.array(train_features)

        for i in range(len(train_features[0])):
            corr = np_utils._corr(train_features[:, i], train_label)
            LogUtil.log('INFO', 'corr(%s_%d)=%f' % (feature_name, i, corr))

        test_features = test_data.apply(Distance.extract_row_compression_dis_ngram, axis=1, raw=True)
        LogUtil.log('INFO', 'extract test features (%s) done' % feature_name)
        Feature.save_dataframe(test_features, test_feature_fp)
        LogUtil.log('INFO', 'save test features (%s) done' % feature_name)

    # @staticmethod
    # def extract_doc2vec_dis(cf, argv):
    #     # 设置参数
    #     feature_name = 'doc2vec_dis'
    #
    #     # 加载数据文件
    #     train_data = pd.read_csv('%s/train.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")
    #     test_data = pd.read_csv('%s/test_with_qid.csv' % cf.get('DEFAULT', 'devel_pt')).fillna(value="")  # [:100]
    #
    #     # 特征存储路径
    #     feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
    #     train_feature_fp = '%s/%s.train.smat' % (feature_pt, feature_name)
    #     test_feature_fp = '%s/%s.test.smat' % (feature_pt, feature_name)
    #
    #     # 抽取特征：train.csv
    #     train_features = train_data.apply(Distance., axis=1, raw=True)
    #     LogUtil.log('INFO', 'extract train features (%s) done' % feature_name)
    #     Feature.save_dataframe(train_features, train_feature_fp)
    #     LogUtil.log('INFO', 'save train features (%s) done' % feature_name)
    #
    #     train_label = train_data['is_duplicate'].values[:]
    #     train_features = train_features.tolist()
    #     train_features = np.array(train_features)
    #
    #     for i in range(len(train_features[0])):
    #         corr = np_utils._corr(train_features[:, i], train_label)
    #         LogUtil.log('INFO', 'corr(%s_%d)=%f' % (feature_name, i, corr))
    #
    #     test_features = test_data.apply(Distance., axis=1, raw=True)
    #     LogUtil.log('INFO', 'extract test features (%s) done' % feature_name)
    #     Feature.save_dataframe(test_features, test_feature_fp)
    #     LogUtil.log('INFO', 'save test features (%s) done' % feature_name)


    @staticmethod
    def run(cf, argv):
        cmd = argv[0]

        if 'extract_jaccard_coef_ngram' == cmd:
            Distance.extract_jaccard_coef_ngram(cf, argv[1:])
        elif 'extract_dice_dis_ngram' == cmd:
            Distance.extract_dice_dis_ngram(cf, argv[1:])
        elif 'extract_edit_dis' == cmd:
            Distance.extract_edit_dis(cf, argv[1:])
        elif 'extract_edit_dis_ngram' == cmd:
            Distance.extract_edit_dis_ngram(cf, argv[1:])
        elif 'extract_compression_dis' == cmd:
            Distance.extract_compression_dis(cf, argv[1:])
        elif 'extract_compression_dis_ngram' == cmd:
            Distance.extract_compression_dis_ngram(cf, argv[1:])
        else:
            LogUtil.log('WARNING', 'NO CMD')


class Predict(object):
    def __init__(self):
        pass

    @staticmethod
    def extract_cv_predict(cf, argv):
        # 传递参数
        version = argv[0]
        cv_num = int(argv[1])
        offline_rawset_name = cf.get('MODEL', 'offline_rawset_name')
        index_fp = cf.get('DEFAULT', 'feature_index_pt')
        # 设置参数
        feature_name = 'cv_predict_%s' % version

        # 加载 offline valid 预测结果
        offline_valid_pred_all_fp = '%s/pred/cv_n%d_valid.%s.pred' % (cf.get('DEFAULT', 'out_pt'), cv_num, offline_rawset_name)
        offline_valid_pred_all_origin = PostProcessor.read_result_list(offline_valid_pred_all_fp)
        # 加载 offline valid 索引
        offline_valid_index_all = []
        for fold_id in range(cv_num):
            offline_valid_indexs_fp = '%s/cv_n%d_f%d_valid.%s.index' % (index_fp, cv_num, fold_id, offline_rawset_name)
            offline_valid_indexs = Feature.load_index(offline_valid_indexs_fp)
            offline_valid_index_all.extend(offline_valid_indexs)
        offline_valid_pred_all = [0] * len(offline_valid_pred_all_origin)
        for index in range(len(offline_valid_index_all)):
            offline_valid_pred_all[offline_valid_index_all[index]] = offline_valid_pred_all_origin[index]

        # 加载 offline test 预测结果
        offline_test_pred_all_fp = '%s/pred/cv_n%d_test.%s.pred' % (
            cf.get('DEFAULT', 'out_pt'), cv_num, offline_rawset_name)
        offline_test_pred_all_origin = PostProcessor.read_result_list(offline_test_pred_all_fp)
        offline_test_pred_all = [0] * len(offline_test_pred_all_origin)
        # 加载 offline test 索引
        offline_test_index_all = []
        for fold_id in range(cv_num):
            offline_test_indexs_fp = '%s/cv_n%d_f%d_test.%s.index' % (
                index_fp, cv_num, fold_id, offline_rawset_name)
            offline_test_indexs = Feature.load_index(offline_test_indexs_fp)
            offline_test_index_all.extend(offline_test_indexs)
        for index in range(len(offline_test_pred_all)):
            offline_test_pred_all[offline_test_index_all[index]] = offline_test_pred_all_origin[index]

        offline_pred_list = [offline_valid_pred_all, offline_test_pred_all]
        offline_pred = PostProcessor.merge_logit_list(offline_pred_list)

        online_pred_fp = '%s/pred/cv_n%d_online.%s.pred' % (cf.get('DEFAULT', 'out_pt'), cv_num, cf.get('MODEL', 'online_test_rawset_name'))
        online_pred = PostProcessor.read_result_list(online_pred_fp)

        offline_pred = [[fv] for fv in offline_pred]
        online_pred = [[fv] for fv in online_pred]

        # 加载数据文件
        train_data = pd.read_csv('%s/train.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")

        # 特征存储路径
        feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
        train_feature_fp = '%s/%s.train.smat' % (feature_pt, feature_name)
        test_feature_fp = '%s/%s.test.smat' % (feature_pt, feature_name)

        # 抽取特征：train.csv
        train_features = sparse.csr_matrix(np.array(offline_pred))
        LogUtil.log('INFO', 'extract train features (%s) done' % feature_name)
        Feature.save_smat(train_features, train_feature_fp)
        LogUtil.log('INFO', 'save train features (%s) done' % feature_name)

        train_label = train_data['is_duplicate'].values[:]
        train_features = train_features.toarray()

        for i in range(len(train_features[0])):
            corr = np_utils._corr(train_features[:, i], train_label)
            LogUtil.log('INFO', 'corr(%s_%d)=%f' % (feature_name, i, corr))

        test_features = sparse.csr_matrix(np.array(online_pred))
        LogUtil.log('INFO', 'extract test features (%s) done' % feature_name)
        Feature.save_smat(test_features, test_feature_fp)
        LogUtil.log('INFO', 'save test features (%s) done' % feature_name)

    @staticmethod
    def extract_cv_predict_new(cf, argv):
        # 传递参数
        version = argv[0]
        cv_num = int(argv[1])
        offline_rawset_name = cf.get('MODEL', 'offline_rawset_name')
        index_fp = cf.get('DEFAULT', 'feature_index_pt')
        # 设置参数
        feature_name = 'cv_predict_new_adj_%s' % version

        # 加载 offline valid 预测结果
        offline_valid_pred_all_fp = '%s/pred/cv_n%d_valid.%s.pred' % (
        cf.get('DEFAULT', 'out_pt'), cv_num, offline_rawset_name)
        offline_valid_pred_all_origin = PostProcessor.read_result_list(offline_valid_pred_all_fp)
        # 加载 offline valid 索引
        offline_valid_index_all = []
        for fold_id in range(cv_num):
            offline_valid_indexs_fp = '%s/cv_n%d_f%d_valid.%s.index' % (index_fp, cv_num, fold_id, offline_rawset_name)
            offline_valid_indexs = Feature.load_index(offline_valid_indexs_fp)
            offline_valid_index_all.extend(offline_valid_indexs)
        offline_valid_pred_all = [0] * len(offline_valid_pred_all_origin)
        for index in range(len(offline_valid_index_all)):
            offline_valid_pred_all[offline_valid_index_all[index]] = offline_valid_pred_all_origin[index]

        # 加载 offline test 预测结果
        offline_test_pred_all_fp = '%s/pred/cv_n%d_test.%s.pred' % (
            cf.get('DEFAULT', 'out_pt'), cv_num, offline_rawset_name)
        offline_test_pred_all_origin = PostProcessor.read_result_list(offline_test_pred_all_fp)
        offline_test_pred_all = [0] * len(offline_test_pred_all_origin)
        # 加载 offline test 索引
        offline_test_index_all = []
        for fold_id in range(cv_num):
            offline_test_indexs_fp = '%s/cv_n%d_f%d_test.%s.index' % (
                index_fp, cv_num, fold_id, offline_rawset_name)
            offline_test_indexs = Feature.load_index(offline_test_indexs_fp)
            offline_test_index_all.extend(offline_test_indexs)
        for index in range(len(offline_test_pred_all)):
            offline_test_pred_all[offline_test_index_all[index]] = offline_test_pred_all_origin[index]

        # 还原scale
        # if cf.get('MODEL', 'has_postprocess') == 'True':
        #     offline_valid_pred_all = [Model.inverse_adj(y) for y in offline_valid_pred_all]
        #     offline_test_pred_all = [Model.inverse_adj(y) for y in offline_test_pred_all]

        offline_pred_list = [offline_valid_pred_all, offline_test_pred_all]
        offline_pred = PostProcessor.merge_logit_list(offline_pred_list)

        # online_pred_fp = '%s/pred/cv_n%d_online.%s.pred' % (
        # cf.get('DEFAULT', 'out_pt'), cv_num, cf.get('MODEL', 'online_test_rawset_name'))
        # online_pred = PostProcessor.read_result_list(online_pred_fp)

        # 加载 online 预测结果
        online_preds = []
        for fold_id in range(cv_num):
            online_pred_fp = '%s/cv_n%d_f%d_online.%s.pred' % (cf.get('DEFAULT', 'pred_pt'), cv_num, fold_id, cf.get('MODEL', 'online_test_rawset_name'))
            online_pred_one = PostProcessor.read_result_list(online_pred_fp)
            # if cf.get('MODEL', 'has_postprocess') == 'True':
            #     online_pred_one = [Model.inverse_adj(y) for y in online_pred_one]
            online_preds.append(online_pred_one)
        # 融合 online 预测结果
        online_pred = []
        for i in range(len(online_preds[0])):
            idd1 = int(random.random() * 5)
            idd2 = (idd1 + 1) % 5
            online_pred.append(PostProcessor.inv_logit(
                np.average(PostProcessor.logit([online_preds[idd1][i], online_preds[idd2][i]]))))

        offline_pred = [[fv] for fv in offline_pred]
        online_pred = [[fv] for fv in online_pred]

        # 加载数据文件
        train_data = pd.read_csv('%s/train.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")

        # 特征存储路径
        feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
        train_feature_fp = '%s/%s.train.smat' % (feature_pt, feature_name)
        test_feature_fp = '%s/%s.test.smat' % (feature_pt, feature_name)

        # 抽取特征：train.csv
        train_features = sparse.csr_matrix(np.array(offline_pred))
        LogUtil.log('INFO', 'extract train features (%s) done' % feature_name)
        Feature.save_smat(train_features, train_feature_fp)
        LogUtil.log('INFO', 'save train features (%s) done' % feature_name)

        train_label = train_data['is_duplicate'].values[:]
        train_features = train_features.toarray()

        for i in range(len(train_features[0])):
            corr = np_utils._corr(train_features[:, i], train_label)
            LogUtil.log('INFO', 'corr(%s_%d)=%f' % (feature_name, i, corr))

        test_features = sparse.csr_matrix(np.array(online_pred))
        LogUtil.log('INFO', 'extract test features (%s) done' % feature_name)
        Feature.save_smat(test_features, test_feature_fp)
        LogUtil.log('INFO', 'save test features (%s) done' % feature_name)


    @staticmethod
    def run(cf, argv):
        cmd = argv[0]

        if 'extract_cv_predict' == cmd:
            Predict.extract_cv_predict(cf, argv[1:])
        elif 'extract_cv_predict_new' == cmd:
            Predict.extract_cv_predict_new(cf, argv[1:])


class Corr(object):
    def __init__(self):
        pass

    @staticmethod
    def cal_corr(cf, argv):
        # 设置参数
        feature_name = argv[0]

        # 特征存储路径
        feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
        train_feature_fp = '%s/%s.train.smat' % (feature_pt, feature_name)
        train_features = Feature.load(train_feature_fp).toarray()

        # 加载数据文件
        train_data = pd.read_csv('%s/train.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")
        train_label = train_data['is_duplicate'].values[:]

        for i in range(len(train_features[0])):
            corr = np_utils._corr(train_features[:, i], train_label)
            LogUtil.log('INFO', 'corr(%s_%d)=%f' % (feature_name, i, corr))

    @staticmethod
    def run(cf, argv):
        cmd = argv[0]

        if 'cal_corr' == cmd:
            Corr.cal_corr(cf, argv[1:])

def print_help():
    print 'extractor <conf_file_fp> -->'
    print '\tword_embedding'
    print '\tid'
    print '\tpostag'
    print '\tdul_num'
    print '\tmath_tag'
    print '\tbtm_vec_cos_sim_dis'
    print '\tPowerfulWordV2'
    print '\tGraph'
    print '\tCount'
    print '\tDistance'
    print '\tCorr'

if __name__ == "__main__":

    if 3 > len(sys.argv):
        print_help()
        exit(1)

    # 读取配置文件
    cf = ConfigParser.ConfigParser()
    cf.read(sys.argv[1])

    cmd = sys.argv[2]
    if 'word_embedding' == cmd:
        WordEmbedding.run(sys.argv[3:])
    elif 'id' == cmd:
        ID.run(sys.argv[3:])
    elif 'postag' == cmd:
        POSTag.run(sys.argv[3:])
    elif 'dul_num' == cmd:
        DulNum.run(sys.argv[3:])
    elif 'math_tag' == cmd:
        MathTag.run(sys.argv[3:])
    elif 'BTMVecCosSimDis' == cmd:
        BTMVecCosSimDis.run(sys.argv[3:])
    elif 'PowerfulWordV2' == cmd:
        PowerfulWordV2.run(cf, sys.argv[3:])
    elif 'Graph' == cmd:
        Graph.run(cf, sys.argv[3:])
    elif 'Count' == cmd:
        Count.run(cf, sys.argv[3:])
    elif 'Distance' == cmd:
        Distance.run(cf, sys.argv[3:])
    elif 'Predict' == cmd:
        Predict.run(cf, sys.argv[3:])
    elif 'Corr' == cmd:
        Corr.run(cf, sys.argv[3:])
    else:
        print_help()
