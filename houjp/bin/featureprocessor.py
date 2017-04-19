# -*- coding: utf-8 -*-
# ! /usr/bin/python

from feature import Feature
import ConfigParser
from os.path import isfile, join
from utils import LogUtil
import sys

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
    def run_gen_feature_swap():
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

    @staticmethod
    def run_gen_feature_with_swap(conf_fp):
        """
        生成线下特征文件，包含swap部分
        :return:
        """
        # 读取配置文件
        cf = ConfigParser.ConfigParser()
        cf.read(conf_fp)
        feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')

        feature_qp_names = Feature.get_feature_names_question_pair(cf)
        rawset_name = 'train'

        for f_name in feature_qp_names:
            feature_fp = '%s/%s.%s.smat' % (feature_pt, f_name, rawset_name)
            feature_swap_fp = '%s/%s.%s_swap.smat' % (feature_pt, f_name, rawset_name)
            feature_with_swap_fp = '%s/%s.%s_with_swap.smat' % (feature_pt, f_name, rawset_name)

            has_with_swap = isfile(feature_with_swap_fp + '.npz')

            if not has_with_swap:
                features = Feature.load(feature_fp)
                features_swap = Feature.load(feature_swap_fp)
                features_with_swap = Feature.merge_row(features, features_swap)
                Feature.save(features_with_swap, feature_with_swap_fp)
                LogUtil.log('INFO', '%s generate with_swap feature done' % f_name)
            else:
                LogUtil.log('INFO', '%s already has with_swap feature' % f_name)


def print_help():
    print 'featureprocessor <conf_file_path> -->'
    print '\tGO'


if __name__ == "__main__":

    if 2 > len(sys.argv):
        print_help()
        exit(1)

    conf_fp = sys.argv[1]

    FeatureProcessor.run_gen_feature_swap()
    FeatureProcessor.run_gen_feature_with_swap(conf_fp)
