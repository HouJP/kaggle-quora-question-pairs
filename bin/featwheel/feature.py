#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/6/19 22:07
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


import hashlib
import random
from os.path import isfile

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

from bin.featwheel.utils import LogUtil


class Feature(object):

    @staticmethod
    def save_feature(feature, feature_file):
        if isinstance(feature, str):
            feature_file.write('%s\n' % feature)
        elif isinstance(feature, list):
            feature = ' '.join(['%s:%s' % (kv[0], kv[1]) for kv in enumerate(feature) if kv[1] != 0])
            feature_file.write('%s\n' % feature)
        else:
            feature_file.write('\n')

    @staticmethod
    def merge_file(feature_pt, feature_name, data_set_name, part_num):
        features = None
        for part_id in range(part_num):
            features_part_fp = '%s/%s.%s.smat.%03d_%03d' % (feature_pt, feature_name, data_set_name, part_num, part_id)
            features_part = Feature.load(features_part_fp)
            if features is None:
                features = features_part
            else:
                features = Feature.merge_row(features, features_part)

        features_fp = '%s/%s.%s.smat' % (feature_pt, feature_name, data_set_name)
        Feature.save_smat(features, features_fp)
        LogUtil.log('INFO',
                    'merge features (%s, %s, %d) done' % (feature_name, data_set_name, part_num))

    @staticmethod
    def sample_row(features, indexs):
        features_sampled = features[indexs, :]
        (row_num, col_num) = features_sampled.shape
        LogUtil.log("INFO", "row sample done, shape=(%d,%d)" % (row_num, col_num))
        return features_sampled

    @staticmethod
    def sample_col(features, indexs):
        features_sampled = features[:, indexs]
        (row_num, col_num) = features_sampled.shape
        LogUtil.log("INFO", "col sample done, shape=(%d,%d)" % (row_num, col_num))
        return features_sampled

    @staticmethod
    def merge_row(feature_1, feature_2):
        """
        merge features made split by row
        :param feature_1: the first part of features
        :param feature_2: the second part of features
        :return: feature matrix
        """
        features = vstack([feature_1, feature_2])
        (row_num, col_num) = features.shape
        LogUtil.log("INFO", "merge row done, shape=(%d,%d)" % (row_num, col_num))
        return features

    @staticmethod
    def merge_col(features_1, features_2):
        """
        merge features made split by column
        :param features_1: the first part of features
        :param features_2: the second part of features
        :return: feature matrix
        """
        features = hstack([features_1, features_2])
        (row_num, col_num) = features.shape
        LogUtil.log("INFO", "merge col done, shape=(%d,%d)" % (row_num, col_num))
        return features

    @staticmethod
    def load(ft_fp):
        """
        WARNING: the NPZ file is buffer files, be careful of these files
        :param ft_fp: features file path
        :return: matrix of features
        """
        has_npz = isfile('%s.npz' % ft_fp)
        if has_npz:
            features = Feature.load_npz(ft_fp)
        else:
            features = Feature.load_smat(ft_fp)
            Feature.save_npz(features, ft_fp)
        return features

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
        save features to disk in binary format
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
        """
        load features from disk, the format:
            row_num col_num
            f1_index:f1_value f2_index:f2_value ...
        """
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
    def save_smat(features, ft_pt):
        """
        save features to disk in SMAT format
        :param features: the matrix of features
        :param ft_pt: features file path
        :return: none
        """
        (row_num, col_num) = features.shape
        data = features.data
        indice = features.indices
        indptr = features.indptr
        f = open(ft_pt, 'w')
        f.write("%d %d\n" % (row_num, col_num))
        ind_indptr = 1
        begin_line = True
        for ind_data in range(len(data)):
            while ind_data == indptr[ind_indptr]:
                f.write('\n')
                begin_line = True
                ind_indptr += 1
            if (data[ind_data] < 1e-12) and (data[ind_data] > -1e-12):
                continue
            if (not begin_line) and (ind_data != indptr[ind_indptr - 1]):
                f.write(' ')
            f.write("%d:%s" % (indice[ind_data], data[ind_data]))
            begin_line = False
        while ind_indptr < len(indptr):
            f.write("\n")
            ind_indptr += 1
        LogUtil.log("INFO", "save smat feature file done (%s)" % ft_pt)
        f.close()

    @staticmethod
    def load_all(feature_pt, feature_names, rawset_name, will_save=False):
        index_begin = 0
        features = None
        for index in reversed(range(1, len(feature_names))):
            f_names_s = '|'.join(feature_names[0:index + 1]) + '|' + rawset_name
            f_names_md5 = hashlib.md5(f_names_s).hexdigest()
            if isfile('%s/md5_%s.smat.npz' % (feature_pt, f_names_md5)):
                index_begin = index
                features = Feature.load('%s/md5_%s.smat' % (feature_pt, f_names_md5))
                break
        LogUtil.log('INFO', 'load %s features [%s, %s)' % (rawset_name, feature_names[0], feature_names[index_begin]))

        if 1 > index_begin:
            features = Feature.load('%s/%s.%s.smat' % (feature_pt, feature_names[0], rawset_name))
        for index in range(index_begin + 1, len(feature_names)):
            features = Feature.merge_col(features,
                                         Feature.load(
                                             '%s/%s.%s.smat' % (feature_pt, feature_names[index], rawset_name)))

        features = features.tocsr()

        if will_save and (index_begin < len(feature_names) - 1):
            f_names_s = '|'.join(feature_names) + '|' + rawset_name
            f_names_md5 = hashlib.md5(f_names_s).hexdigest()
            Feature.save_npz(features, '%s/md5_%s.smat' % (feature_pt, f_names_md5))
        return features

    @staticmethod
    def balance_index(indexs, labels, positive_rate):
        """
        balance indexs to adjust the positive rate
        :param indexs: index vector to sample raw data set
        :param labels: label vector of raw data set
        :param positive_rate: positive rate
        :return: index vector after balanced
        """
        if positive_rate < 1e-6 or positive_rate > 1. - 1e-6:
            return indexs
        pos_indexs = [index for index in indexs if labels[index] == 1.]
        neg_indexs = [index for index in indexs if labels[index] == 0.]
        origin_rate = 1.0 * len(pos_indexs) / len(indexs)
        LogUtil.log("INFO", "original: len(pos)=%d, len(neg)=%d, rate=%.2f%%" % (
            len(pos_indexs), len(neg_indexs), 100.0 * origin_rate))
        if origin_rate < positive_rate:
            pos_indexs, neg_indexs = neg_indexs, pos_indexs
            origin_rate = 1.0 - origin_rate
            positive_rate = 1.0 - positive_rate
            LogUtil.log("INFO", "increase postive instances ...")
        else:
            LogUtil.log("INFO", "increase negtive instances ...")
        k = (1. - positive_rate) * origin_rate / positive_rate / (1 - origin_rate)
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
