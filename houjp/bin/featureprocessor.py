# -*- coding: utf-8 -*-
# ! /usr/bin/python

from feature import Feature
import ConfigParser
from os.path import isfile, join
from utils import LogUtil

class FeatureProcessor(object):
    """
    特征处理工具
    """

    def __init__(self):
        return

    @staticmethod
    def swap_feature(feature_pt, feature_name, feature_index, rawset_name):
        """
        特征重排列
        :param feature_pt:
        :param feature_name:
        :param rawset_name:
        :return:
        """
        feature_fp = '%s/%s.%s.smat' % (feature_pt, feature_name, rawset_name)
        feature_swap_fp = '%s/%s.%s_swap.smat' % (feature_pt, feature_name, rawset_name)

        has_swap = isfile(feature_swap_fp + '.npz')
        if not has_swap:
            features = Feature.load(feature_fp)
            features_swap = Feature.sample_col(features, feature_index)
            Feature.save(features_swap, feature_swap_fp)
            LogUtil.log('INFO', '%s generate swap feature done' % feature_name)
        else:
            LogUtil.log('INFO', '%s already has swap feature' % feature_name)

    @staticmethod
    def load_feature_swap_conf(conf_fp):
        """
        加载配置文件
        :return:
        """
        f_names = []
        f_indexs = []
        f = open(conf_fp, 'r')
        for line in f:
            [f_name, f_index_s] = line.strip().split('\t')
            f_names.append(f_name)
            f_index_subs_s = f_index_s.split(',')
            f_index = []
            for f_index_sub in f_index_subs_s:
                if ':' in f_index_sub:
                    [b, e] = f_index_sub.split(':')
                    f_index += range(int(b.strip()), int(e.strip()))
                else:
                    f_index.append(int(f_index_sub.strip()))
            f_indexs.append(f_index)
        f.close()
        return f_names, f_indexs

    @staticmethod
    def run_feature_swap():
        """
        交换<Q1,Q2>特征
        :return:
        """
        # 读取配置文件
        cf = ConfigParser.ConfigParser()
        cf.read("../conf/python.conf")
        feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')

        # 加载配置文件
        feature_swap_conf_fp = '../conf/feature_swap.conf'
        f_names, f_indexs = FeatureProcessor.load_feature_swap_conf(feature_swap_conf_fp)

        # 特征变换
        for i in range(len(f_names)):
            f_name = f_names[i]
            f_index = f_indexs[i]

            rawset_name = 'train'
            FeatureProcessor.swap_feature(feature_pt, f_name, f_index, rawset_name)

            rawset_name = 'test'
            FeatureProcessor.swap_feature(feature_pt, f_name, f_index, rawset_name)


if __name__ == "__main__":
    FeatureProcessor.run_feature_swap()
