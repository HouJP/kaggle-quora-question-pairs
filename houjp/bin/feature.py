#! /usr/bin/python
# -*- coding: utf-8 -*-

import ConfigParser
from scipy.sparse import csr_matrix, hstack
from nltk.corpus import stopwords
from utils import LogUtil
import random
import numpy as np
from os import listdir
from os.path import isfile, join
import re

from utils import DataUtil


class Feature(object):
    '''
    特征工程工具
    '''

    # 停用词
    stops = set(stopwords.words("english"))
    # train.csv中IDF字典
    train_idf = {}

    def __init__(self):
        return

    @staticmethod
    def load_npz(ft_fp):
        loader = np.load('%s.npz' % ft_fp)
        features = csr_matrix((loader['data'],
                           loader['indices'],
                           loader['indptr']),
                          shape=loader['shape'])
        LogUtil.log("INFO", "load npz feature file done (%s)" % ft_fp)
        return features

    @staticmethod
    def save_npz(features, ft_fp):
        """
        存储二进制特征文件
        :param features:
        :param ft_fp:
        :return:
        """
        np.savez(ft_fp,
                 data=features.data,
                 indices=features.indices,
                 indptr=features.indptr,
                 shape=features.shape)
        LogUtil.log('INFO', 'save npz feature file done (%s)' % ft_fp)

    @staticmethod
    def load_smat(ft_fp):
        '''
        加载特征文件，特征文件格式如下：
        row_num col_num
        f1_index:f1_value f2_index:f2_value ...
        '''
        data = []
        indice = []
        indptr = [0]
        f = open(ft_fp)
        [row_num, col_num] = [int(num) for num in f.readline().strip().split()]
        for line in f:
            line = line.strip()
            subs = line.split()
            for sub in subs:
                [f_index, f_value] = sub.split(":")
                f_index = int(f_index)
                f_value = float(f_value)
                data.append(f_value)
                indice.append(f_index)
            indptr.append(len(data))
        f.close()
        features = csr_matrix((data, indice, indptr), shape=(row_num, col_num), dtype=float)
        LogUtil.log("INFO", "load smat feature file done (%s)" % ft_fp)
        return features

    @staticmethod
    def load_with_part_id(ft_fp, id_part, n_line):
        ft_id_fp = '%s.%02d' % (ft_fp, id_part)
        has_part = isfile(ft_id_fp)
        features = None
        if has_part:
            features = Feature.load(ft_id_fp)
        else:
            Feature.split_feature(ft_fp, n_line)
            features = Feature.load(ft_id_fp)
        return features

    @staticmethod
    def load(ft_fp):
        """
        WARNING: 很容易造成smat格式与npz格式文件内容不一致
        :param ft_fp:
        :return:
        """
        has_npz = isfile('%s.npz' % ft_fp)
        features = None
        if has_npz:
            features = Feature.load_npz(ft_fp)
        else:
            features = Feature.load_smat(ft_fp)
            Feature.save_npz(features, ft_fp)
        return features

    @staticmethod
    def split_feature(ft_fp, n_line):
        features = Feature.load('%s' % ft_fp)
        index_start = 0
        while index_start < features.shape[0]:
            index_end = min(index_start + n_line, features.shape[0])
            sub_features = Feature.sample_with_index(features, index_start, index_end)
            Feature.save(sub_features, '%s.%02d' % (ft_fp, index_start / n_line))
            index_start += n_line

    @staticmethod
    def load_all_features_with_part_id(cf, rawset_name, id_part):
        """
        加载部分数据全部特征
        :param cf:
        :param rawset_name:
        :param id_part:
        :return:
        """
        # 加载<Q1,Q2>二元组特征
        n_line = cf.getint('MODEL', 'n_line')
        feature_qp_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
        feature_qp_names = Feature.get_feature_names_question_pair(cf)
        features = Feature.load_mul_features_with_part_id(feature_qp_pt,
                                                          feature_qp_names,
                                                          rawset_name,
                                                          id_part,
                                                          n_line)
        # 加载<Question>特征
        # TODO
        return features

    @staticmethod
    def load_mul_features_with_part_id(feature_pt, feature_names, rawset_name, id_part, n_line):
        features = Feature.load_with_part_id('%s/%s.%s.smat' % (feature_pt, feature_names[0], rawset_name), id_part, n_line)
        for index in range(1, len(feature_names)):
            features = Feature.merge(features,
                                     Feature.load_with_part_id('%s/%s.%s.smat' % (feature_pt,
                                                                                  feature_names[index],
                                                                                  rawset_name), id_part, n_line))
        return features

    @staticmethod
    def load_all_features(cf, rawset_name):
        '''
        加载全部特征矩阵
        '''
        # 加载<Q1,Q2>二元组特征
        feature_qp_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
        feature_qp_names = Feature.get_feature_names_question_pair(cf)
        features = Feature.load_mul_features(feature_qp_pt, feature_qp_names, rawset_name)
        # 加载<Question>特征
        # TODO
        return features

    @staticmethod
    def load_mul_features(feature_pt, feature_names, rawset_name):
        features = Feature.load('%s/%s.%s.smat' % (feature_pt, feature_names[0], rawset_name))
        for index in range(1, len(feature_names)):
            features = Feature.merge(features,
                                     Feature.load('%s/%s.%s.smat' % (feature_pt, feature_names[index], rawset_name)))
        return features

    @staticmethod
    def save_smat(features, ft_pt):
        '''
        存储特征文件
        '''
        (row_num, col_num) = features.shape
        data = features.data
        indice = features.indices
        indptr = features.indptr
        f = open(ft_pt, 'w')
        f.write("%d %d\n" % (row_num, col_num))
        ind_indptr = 1
        blank_line = True
        for ind_data in range(len(data)):
            while ind_data == indptr[ind_indptr]:
                if blank_line:
                    f.write('0:0')
                f.write('\n')
                blank_line = True
                ind_indptr += 1
            if ind_data != indptr[ind_indptr - 1]:
                f.write(' ')
            f.write("%d:%f" % (indice[ind_data], data[ind_data]))
            blank_line = False
        f.write("\n")
        LogUtil.log("INFO", "save smat feature file done (%s)" % ft_pt)
        f.close()

    @staticmethod
    def save(features, ft_fp):
        Feature.save_npz(features, ft_fp)
        # Feature.save_smat(features, ft_fp)

    @staticmethod
    def save_dataframe(features, ft_pt):
        '''
        存储DataFrame特征文件
        '''
        features = np.array(features)
        f = open(ft_pt, 'w')
        f.write('%d %d\n' % (len(features), len(features[0])))
        for row in features:
            for ind in range(len(row)):
                f.write('%d:%f' % (ind, float(row[ind])))
                if ind < len(row) - 1:
                    f.write(' ')
                else:
                    f.write('\n')
        f.close()
        LogUtil.log("INFO", "save dataframe feature done (%s)" % ft_pt)
        return

    @staticmethod
    def merge(features_1, features_2):
        '''
        横向合并特征矩阵，即为每个实例增加特征
        '''
        features = hstack([features_1, features_2])
        (row_num, col_num) = features.shape
        LogUtil.log("INFO", "merge feature done, shape=(%d,%d)" % (row_num, col_num))
        return features.tocsr()

    @staticmethod
    def get_feature_names_question(cf):
        '''
        获取针对<问题>的特征池中的特证名
        '''
        return cf.get('FEATURE', 'feature_names_question').split()

    @staticmethod
    def get_feature_names_question_pair(cf):
        '''
        获取针对<问题，问题>二元组的特征池中的特征名
        '''
        return cf.get('FEATURE', 'feature_names_question_pair').split()

    @staticmethod
    def sample_with_index(features, row_begin, row_end):
        """
        根据索引对特征矩阵切片
        :param features:
        :param row_begin:
        :param row_end:
        :return:
        """
        features_sampled = features[row_begin : row_end, :]
        (row_num, col_num) = features_sampled.shape
        LogUtil.log("INFO", "sample feature done, shape=(%d,%d)" % (row_num, col_num))
        return features_sampled

    @staticmethod
    def sample_with_index(features, indexs):
        '''
        根据索引采样特征向量
        '''
        features_sampled = features[indexs, :]
        (row_num, col_num) = features_sampled.shape
        LogUtil.log("INFO", "sample feature done, shape=(%d,%d)" % (row_num, col_num))
        return features_sampled

    @staticmethod
    def load_index(fp):
        '''
        加载特征索引文件
        '''
        f = open(fp)
        indexs = [int(line) for line in f.readlines()]
        LogUtil.log("INFO", "load index done, len(index)=%d" % (len(indexs)))
        f.close()
        return indexs

    @staticmethod
    def balance_index(indexs, labels, rate):
        '''
        增加正样本或者负样本的比例，使得正样本的比例在rate附近
        '''
        if (rate < 1e-6 or rate > 1. - 1e-6):
            return indexs
        pos_indexs = [index for index in indexs if labels[index] == 1.]
        neg_indexs = [index for index in indexs if labels[index] == 0.]
        origin_rate = 1.0 * len(pos_indexs) / len(indexs)
        LogUtil.log("INFO", "original: len(pos)=%d, len(neg)=%d, rate=%.2f%%" % (
        len(pos_indexs), len(neg_indexs), 100.0 * origin_rate))
        if (origin_rate < rate):
            # 始终采样负样本
            pos_indexs, neg_indexs = neg_indexs, pos_indexs
            origin_rate = 1.0 - origin_rate
            rate = 1.0 - rate
            LogUtil.log("INFO", "increase postive instances ...")
        else:
            LogUtil.log("INFO", "increase negtive instances ...")
        k = 3.  # (1. - rate) * origin_rate / rate / (1 - origin_rate)
        LogUtil.log("INFO", "k=%.4f" % k)
        balance_indexs = pos_indexs
        while k > 1e-6:
            if k > 1. - 1e-6:
                balance_indexs.extend(neg_indexs)
            else:
                balance_indexs.extend(random.sample(neg_indexs, int(k * len(neg_indexs))))
            k -= 1.
        pos_indexs = [index for index in balance_indexs if labels[index] == 1.]
        neg_indexs = [index for index in balance_indexs if labels[index] == 0.]
        balanced_rate = 1.0 * len(pos_indexs) / len(balance_indexs)
        LogUtil.log("INFO", "balanced: len(pos)=%d, len(neg)=%d, rate=%.2f%%" % (
        len(pos_indexs), len(neg_indexs), 100.0 * balanced_rate))
        return balance_indexs

    @staticmethod
    def demo():
        '''
        使用样例代码
        '''
        # 读取配置文件
        cf = ConfigParser.ConfigParser()
        cf.read("../conf/python.conf")

        # 加载特征文件
        features = Feature.load("%s/feature1.demo.smat" % cf.get('DEFAULT', 'feature_question_pt'))
        # 存储特征文件
        Feature.save(features, "%s/feature2.demo.smat" % cf.get('DEFAULT', 'feature_question_pt'))
        # 合并特征
        Feature.merge(features, features)
        # 获取<问题>特征池中的特征名
        Feature.get_feature_names_question(cf)
        # 加载索引文件
        indexs = Feature.load_index("%s/vali.demo.index" % cf.get('DEFAULT', 'feature_index_pt'))
        # 根据索引对特征采样
        features = Feature.sample_with_index(features, indexs)
        # 正负样本均衡化
        rate = 0.165
        train311_train_indexs_fp = '%s/train_311.train.index' % cf.get('DEFAULT', 'feature_index_pt')
        train311_train_indexs = Feature.load_index(train311_train_indexs_fp)
        train_labels_fp = '%s/train.label' % cf.get('DEFAULT', 'feature_label_pt')
        train_labels = DataUtil.load_vector(train_labels_fp, True)
        balanced_indexs = Feature.balance_index(train311_train_indexs, train_labels, rate)

    @staticmethod
    def test():
        '''
        测试函数
        '''
        # 读取配置文件
        cf = ConfigParser.ConfigParser()
        cf.read("../conf/python.conf")

        # split all features
        features = Feature.load('/Users/houjianpeng/Github/kaggle-quora-question-pairs/data/feature/question/feature1.demo.smat')
        Feature.save(features, '/Users/houjianpeng/Github/kaggle-quora-question-pairs/data/feature/question/feature2.demo.smat')
        # Feature.split_all_features(cf)

if __name__ == "__main__":
    Feature.test()
