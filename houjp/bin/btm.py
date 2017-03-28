#! /usr/bin/python
# -*- coding: utf-8 -*-

import ConfigParser
import pandas as pd
from utils import DataUtil
import nltk


class BTM(object):

    @staticmethod
    def get_qid2question(df):
        qid2question = {}
        for index, row in df.iterrows():
            q1 = nltk.word_tokenize(str(row['question1']).lower().decode('utf-8'))
            q2 = nltk.word_tokenize(str(row['question2']).lower().decode('utf-8'))
            qid1 = str(row['qid1'])
            qid2 = str(row['qid2'])
            qid2question[qid1] = ' '.join(q1)
            qid2question[qid2] = ' '.join(q2)
        return qid2question

    @staticmethod
    def get_all_qid2question(train_df, test_df):
        """
        获取全部
        :param train_df:
        :param test_df:
        :return:
        """
        train_qid2question = BTM.get_qid2question(train_df)
        test_qid2question = BTM.get_qid2question(test_df)
        all_qid2question = dict(train_qid2question, **test_qid2question)
        return all_qid2question

    @staticmethod
    def save_all_qid2question():
        # 读取配置文件
        cf = ConfigParser.ConfigParser()
        cf.read("../conf/python.conf")

        # 加载train.csv文件
        train_data = pd.read_csv('%s/train.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")  # [:100]
        # 加载test.csv文件
        test_data = pd.read_csv('%s/test_with_qid.csv' % cf.get('DEFAULT', 'devel_pt')).fillna(value="")  # [:100]

        # 存储索引文件
        qid2question_qid_fp = '%s/qid2question.all.qid' % cf.get('DEFAULT', 'devel_pt')
        qid2question_question_fp = '%s/qid2question.all.question' % cf.get('DEFAULT', 'devel_pt')

        # 获取qid2question
        all_qid2question = BTM.get_all_qid2question(train_data, test_data)

        all_qid = []
        all_question = []
        for qid in all_qid2question:
            all_qid.append(qid)
            all_question.append(all_qid2question[qid])

        # 存储索引
        DataUtil.save_vector(qid2question_qid_fp, all_qid, 'w')
        DataUtil.save_vector(qid2question_qid_fp, all_question, 'w')


if __name__ == "__main__":
    # 存储qid和question文件
    BTM.save_all_qid2question()

