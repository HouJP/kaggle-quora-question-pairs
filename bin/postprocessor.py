# -*- coding: utf-8 -*-
# ! /usr/bin/python

import numpy as np
import getopt
import sys
from featwheel.utils import DataUtil
from featwheel.feature import Feature

class PostProcessor(object):
    @staticmethod
    def adj(x, te, tr):
        a = te / tr
        b = (1 - te) / (1 - tr)
        return a * x / (a * x + b * (1 - x))

    @staticmethod
    def rescale(config, online_preds_fp):
        online_preds = DataUtil.load_vector(online_preds_fp, 'float')

        feature_name = 'graph_edge_max_clique_size'
        feature_pt = config.get('DEFAULT', 'feature_pt')
        test_feature_fp = '%s/%s.test.smat' % (feature_pt, feature_name)
        test_features_mc = Feature.load(test_feature_fp).toarray()

        feature_name = 'graph_edge_cc_size'
        feature_pt = config.get('DEFAULT', 'feature_pt')
        test_feature_fp = '%s/%s.test.smat' % (feature_pt, feature_name)
        test_features_cc = Feature.load(test_feature_fp).toarray()

        for index in range(len(online_preds)):
            score = online_preds[index]
            if test_features_mc[index][0] == 3.:
                score = PostProcessor.adj(score, te=0.40883512, tr=0.623191)
            elif test_features_mc[index][0] > 3.:
                score = PostProcessor.adj(score, te=0.96503024, tr=0.972554)
            else:
                if test_features_cc[index][0] < 3.:
                    score = PostProcessor.adj(score, te=0.05739666, tr=0.233473)
                else:
                    score = PostProcessor.adj(score, te=0.04503431, tr=0.149471)
            online_preds[index] = score

        DataUtil.save_vector(online_preds_fp + '.rescale', online_preds)


