#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/6/21 12:26
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


import ConfigParser
import random

import numpy as np
from scipy import sparse

from bin.featwheel.utils import LogUtil
from ..featwheel.feature import Feature
from ..postprocessor import PostProcessor


class Stacking(object):

    def __init__(self, config_fp):
        # load configuration file
        self.config = ConfigParser.ConfigParser()
        self.config.read(config_fp)

    def extract(self):
        version = self.config.get('INFO', 'version')
        cv_num = self.config.get('INFO', 'cv_num')
        offline_rawset_name = self.config.get('MODEL', 'offline_rawset_name')
        index_fp = self.config.get('DIRECTORY', 'feature_pt')
        feature_name = '%s_%s' % (self.__class__.__name__, version)

        # load prediction of offline tests
        offline_test_pred_all_fp = '%s/pred/cv_n%d_test.%s.pred' % (
            self.config.get('DIRECTORY', 'out_pt'), cv_num, offline_rawset_name)
        offline_test_pred_all_origin = PostProcessor.read_result_list(offline_test_pred_all_fp)
        offline_test_pred_all = [0] * len(offline_test_pred_all_origin)
        # load index of offline tests
        offline_test_index_all = list()
        for fold_id in range(cv_num):
            offline_test_indexs_fp = '%s/cv_n%d_f%d_test.%s.index' % (
                index_fp, cv_num, fold_id, offline_rawset_name)
            offline_test_indexs = Feature.load_index(offline_test_indexs_fp)
            offline_test_index_all.extend(offline_test_indexs)
        for index in range(len(offline_test_pred_all)):
            offline_test_pred_all[offline_test_index_all[index]] = offline_test_pred_all_origin[index]

        # load prediction of online data set
        online_preds = list()
        for fold_id in range(cv_num):
            online_pred_fp = '%s/cv_n%d_f%d_online.%s.pred' % (
                self.config.get('DIRECTORY', 'pred_pt'),
                cv_num,
                fold_id,
                self.config.get('MODEL', 'online_test_rawset_name'))
            online_pred_one = PostProcessor.read_result_list(online_pred_fp)
            online_preds.append(online_pred_one)
        # sample for online prediction
        online_pred = []
        for i in range(len(online_preds[0])):
            cv_id = int(random.random() * cv_num)
            online_pred.append(online_preds[cv_id][i])

        offline_pred = [[fv] for fv in offline_test_pred_all]
        online_pred = [[fv] for fv in online_pred]

        # directory of features
        feature_pt = self.config.get('DIRECTORY', 'feature_pt')
        train_feature_fp = '%s/%s.train.smat' % (feature_pt, feature_name)
        test_feature_fp = '%s/%s.test.smat' % (feature_pt, feature_name)

        train_features = sparse.csr_matrix(np.array(offline_pred))
        Feature.save_smat(train_features, train_feature_fp)
        LogUtil.log('INFO', 'save train features (%s) done' % feature_name)

        test_features = sparse.csr_matrix(np.array(online_pred))
        Feature.save_smat(test_features, test_feature_fp)
        LogUtil.log('INFO', 'save test features (%s) done' % feature_name)