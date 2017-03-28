# -*- coding: utf-8 -*-
# ! /usr/bin/python

import sys
import numpy as np
import pandas as pd
import ConfigParser
from utils import DataUtil
from utils import LogUtil

class Preprocessor(object):
    '''
    预处理工具
    '''

    def __init__(self):
        return

    @staticmethod
    def get_qid2question(df):
        '''
        获取Map(qid, question)
        '''
        qid2question = {}
        qids = df['qid1'].tolist() + df['qid2'].tolist()
        questions = df['question1'].tolist() + df['question2'].tolist()
        for ind in range(len(qids)):
            qid2question[qids[ind]] = questions[ind]
        LogUtil.log("INFO", "len(qids)=%d, len(unique_qids)=%d" % (len(qids), len(qid2question)))
        return qid2question

    @staticmethod
    def get_labels(df):
        '''
        获取标签
        '''
        labels = df['is_duplicate'].tolist()
        LogUtil.log("INFO", "num(1)=%d, num(0)=%d" % (sum(labels), len(labels) - sum(labels)))
        return labels

    @staticmethod
    def get_test_ids(df):
        '''
        获取test_id列表
        '''
        ids = df['test_id'].tolist()
        LogUtil.log("INFO", "len(ids)=%d" % len(ids))
        return ids

    @staticmethod
    def static_dul_question(df):
        '''
        统计重复语句
        '''
        questions = df['question1'].tolist() + df['question2'].tolist()
        len_questions = len(questions)
        len_uniq_questions = len(set(questions))
        LogUtil.log("INFO", "len(questions)=%d, len(unique_questions)=%d, rate=%f" % (
        len_questions, len_uniq_questions, 1.0 * len_uniq_questions / len_questions))

    @staticmethod
    def add_qid_for_test(df):
        """
        增加qid1, qid2
        :param df:
        :return:
        """
        df['qid1'] = df.apply(lambda r: ('T%08d' % (2 * r.test_id)), axis=1, raw=True)
        df['qid2'] = df.apply(lambda r: ('T%08d' % (2 * r.test_id + 1)), axis=1, raw=True)
        return df


class PreprocessorRunner(object):
    '''
    预处理业务
    '''

    def __init__(self):
        pass

    @staticmethod
    def get_qid2question(cf):
        '''
        获取train.csv和test.csv的Map(qid, question)
        '''
        train_df = pd.read_csv('%s/train.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")
        train_qid2question = Preprocessor.get_qid2question(train_df)
        qid2question_fp = '%s/train_qid2question.csv' % cf.get('DEFAULT', 'devel_pt')
        DataUtil.save_dic2csv(train_qid2question, '"qid","question"', qid2question_fp)

    @staticmethod
    def get_labels(cf):
        '''
        获取train.csv中标签（is_duplicate）信息，并存储
        '''
        train_df = pd.read_csv('%s/train.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")
        train_labels = Preprocessor.get_labels(train_df)
        train_labels_fp = '%s/train.label' % cf.get('DEFAULT', 'feature_label_pt')
        DataUtil.save_vector(train_labels_fp, train_labels, 'w')
        LogUtil.log("INFO", "save label file done (%s)" % train_labels_fp)

    @staticmethod
    def static_dul_question(cf):
        '''
        统计重复语句
        '''
        train_df = pd.read_csv('%s/train.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")
        Preprocessor.static_dul_question(train_df)
        test_df = pd.read_csv('%s/test.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")
        Preprocessor.static_dul_question(test_df)

    @staticmethod
    def get_test_ids(cf):
        '''
        存储test.csv中test_id列表
        '''
        test_df = pd.read_csv('%s/test.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")
        test_ids = Preprocessor.get_test_ids(test_df)
        test_ids_fp = '%s/test.id' % cf.get('DEFAULT', 'feature_id_pt')
        DataUtil.save_vector(test_ids_fp, test_ids, 'w')
        LogUtil.log("INFO", "save test id file done (%s)" % test_ids_fp)

    @staticmethod
    def get_test_indexs(cf):
        '''
        存储test.csv索引文件
        '''
        test_df = pd.read_csv('%s/test.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")
        test_indexs_fp = '%s/full.test.index' % cf.get('DEFAULT', 'feature_index_pt')
        DataUtil.save_vector(test_indexs_fp, range(len(test_df)), 'w')
        LogUtil.log("INFO", "save test index file done (%s)" % test_indexs_fp)

    @staticmethod
    def get_test_labels(cf):
        '''
        存储test.csv标签文件
        '''
        test_df = pd.read_csv('%s/test.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")
        test_labels_fp = '%s/test.label' % cf.get('DEFAULT', 'feature_label_pt')
        DataUtil.save_vector(test_labels_fp, [0 for i in range(len(test_df))], 'w')
        LogUtil.log("INFO", "save test labels file done (%s)" % test_labels_fp)

    @staticmethod
    def add_qid_for_test(cf):
        """
        为test.csv增加qid
        :param cf:
        :return:
        """
        test_df = pd.read_csv('%s/test.csv' % cf.get('DEFAULT', 'origin_pt'))
        LogUtil.log('INFO', 'load test dataframe done')
        test_df = Preprocessor.add_qid_for_test(test_df)
        LogUtil.log('INFO', 'add qid for test dataframe done')
        test_df.to_csv('%s/test_with_qid.csv' % cf.get('DEFAULT', 'devel_pt'))
        LogUtil.log('INFO', 'save test dataframe with qid done')


if __name__ == "__main__":
    # 读取配置文件
    cf = ConfigParser.ConfigParser()
    cf.read("../conf/python.conf")

    # PreprocessorRunner.get_qid2question(cf)
    # PreprocessorRunner.static_dul_question(cf)
    # PreprocessorRunner.get_labels(cf)
    # PreprocessorRunner.get_test_ids(cf)
    # PreprocessorRunner.get_test_indexs(cf)
    # PreprocessorRunner.get_test_labels(cf)
    PreprocessorRunner.add_qid_for_test(cf)
