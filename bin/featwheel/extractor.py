#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/6/16 10:55
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


import ConfigParser

import pandas as pd

from bin.featwheel.utils import LogUtil
from feature import Feature


class Extractor(object):

    def __init__(self, config_fp):
        # set feature name
        self.feature_name = self.__class__.__name__
        # set feature file path
        self.data_feature_fp = None
        # load configuration file
        self.config = ConfigParser.ConfigParser()
        self.config.read(config_fp)

    def get_feature_num(self):
        assert False, 'Please override function: Extractor.get_feature_num()'

    def extract_row(self, row):
        assert False, 'Please override function: Extractor.extract_row()'

    def extract(self, data_set_name, part_num=1, part_id=0):
        """
        Extract the feature from original data set
        :param data_set_name: name of data set
        :param part_num: number of partitions of data
        :param part_id: partition ID which will be extracted
        :return:
        """
        # load data set from disk
        data = pd.read_csv('%s/%s.csv' % (self.config.get('DEFAULT', 'source_pt'), data_set_name)).fillna(value="")
        begin_id = int(1. * len(data) / part_num * part_id)
        end_id = int(1. * len(data) / part_num * (part_id + 1))

        # set feature file path
        feature_pt = self.config.get('DEFAULT', 'feature_pt')
        if 1 == part_num:
            self.data_feature_fp = '%s/%s.%s.smat' % (feature_pt, self.feature_name, data_set_name)
        else:
            self.data_feature_fp = '%s/%s.%s.smat.%03d_%03d' % (feature_pt,
                                                                self.feature_name,
                                                                data_set_name,
                                                                part_num,
                                                                part_id)

        feature_file = open(self.data_feature_fp, 'w')
        feature_file.write('%d %d\n' % (end_id - begin_id, int(self.get_feature_num())))
        # extract feature
        for index, row in data[begin_id:end_id].iterrows():
            feature = self.extract_row(row)
            Feature.save_feature(feature, feature_file)
        feature_file.close()

        LogUtil.log('INFO',
                    'save features (%s, %s, %d, %d) done' % (self.feature_name, data_set_name, part_num, part_id))
