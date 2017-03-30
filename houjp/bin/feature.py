#! /usr/bin/python
# -*- coding: utf-8 -*-

import ConfigParser
from scipy.sparse import csr_matrix, hstack
from nltk.corpus import stopwords
from utils import LogUtil
import random
import numpy as np

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
    def load(ft_pt):
        '''
        加载特征文件，特征文件格式如下：
        row_num col_num
        f1_index:f1_value f2_index:f2_value ...
        '''
        data = []
        indice = []
        indptr = [0]
        f = open(ft_pt)
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
        LogUtil.log("INFO", "load feature file done (%s)" % ft_pt)
        return csr_matrix((data, indice, indptr), shape=(row_num, col_num), dtype=float)

    @staticmethod
    def load_all_features(cf, rawset_name, id_part):
        '''
        加载全部特征矩阵
        '''
        # 加载<Q1,Q2>二元组特征
        feature_qp_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
        feature_qp_names = Feature.get_feature_names_question_pair(cf)
        features = Feature.load_mul_features(feature_qp_pt, feature_qp_names, rawset_name, id_part)
        # 加载<Question>特征
        # TODO
        return features

    @staticmethod
    def load_mul_features(feature_pt, feature_names, rawset_name, id_part):
        features = Feature.load('%s/%s.%s.smat.%02d' % (feature_pt, feature_names[0], rawset_name, id_part))
        for index in range(1, len(feature_names)):
            features = Feature.merge(features,
                                     Feature.load('%s/%s.%s.smat.%02d' % (feature_pt, feature_names[index], rawset_name, id_part)))
        return features

    @staticmethod
    def save(features, ft_pt):
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
        for ind_data in range(len(data)):
            f.write("%d:%f" % (indice[ind_data], data[ind_data]))
            if (ind_data < indptr[ind_indptr] - 1):
                f.write(" ")
            else:
                f.write("\n")
                ind_indptr += 1
        LogUtil.log("INFO", "save feature file done (%s)" % ft_pt)
        f.close()

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


if __name__ == "__main__":
    Feature.demo()
