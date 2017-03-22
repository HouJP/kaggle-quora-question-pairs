#! /usr/bin/python
# -*- coding: utf-8 -*-

import ConfigParser
from nltk.corpus import stopwords
from feature import Feature
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import numpy as np


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


class TFIDWordMatchShare(object):
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
        TFIDWordMatchShare.weights = {word: TFIDWordMatchShare.cal_weight(count) for word, count in counts.items()}

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
            if word not in TFIDWordMatchShare.stops:
                q1words[word] = 1
        for word in str(row['question2']).lower().split():
            if word not in TFIDWordMatchShare.stops:
                q2words[word] = 1
        if len(q1words) == 0 or len(q2words) == 0:
            # The computer-generated chaff includes a few questions that are nothing but stopwords
            return [0.]

        shared_weights = [TFIDWordMatchShare.weights.get(w, 0) for w in q1words.keys() if w in q2words] + [TFIDWordMatchShare.weights.get(w, 0) for w in                                                                                q2words.keys() if w in q1words]
        total_weights = [TFIDWordMatchShare.weights.get(w, 0) for w in q1words] + [TFIDWordMatchShare.weights.get(w, 0) for w in q2words]
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
        TFIDWordMatchShare.get_weights(train_data)
        # 抽取特征
        train_features = train_df.apply(TFIDWordMatchShare.tfidf_word_match_share, axis=1, raw=True)
        Feature.save_dataframe(train_features, feature_pt + '/tfidf_word_match_share.train.smat')
        test_features = test_df.apply(TFIDWordMatchShare.tfidf_word_match_share, axis=1, raw=True)
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


if __name__ == "__main__":
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
    # WordMatchShare.run(train_data, test_data, feature_path)
    TFIDWordMatchShare.run(train_data, test_data, feature_path)