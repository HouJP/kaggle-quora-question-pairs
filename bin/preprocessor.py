#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/6/15 15:32
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


import ConfigParser
import random
import re

import nltk
import pandas as pd
from nltk.stem import SnowballStemmer

from bin.featwheel.utils import DataUtil, LogUtil


class TextPreProcessor(object):

    _stemmer = SnowballStemmer('english')

    def __init__(self):
        pass

    @staticmethod
    def clean_text(text):
        """
        Clean text
        :param text: the string of text
        :return: text string after cleaning
        """
        # unit
        text = re.sub(r"(\d+)kgs ", lambda m: m.group(1) + ' kg ', text)        # e.g. 4kgs => 4 kg
        text = re.sub(r"(\d+)kg ", lambda m: m.group(1) + ' kg ', text)         # e.g. 4kg => 4 kg
        text = re.sub(r"(\d+)k ", lambda m: m.group(1) + '000 ', text)          # e.g. 4k => 4000
        text = re.sub(r"\$(\d+)", lambda m: m.group(1) + ' dollar ', text)
        text = re.sub(r"(\d+)\$", lambda m: m.group(1) + ' dollar ', text)

        # acronym
        text = re.sub(r"can\'t", "can not", text)
        text = re.sub(r"cannot", "can not ", text)
        text = re.sub(r"what\'s", "what is", text)
        text = re.sub(r"What\'s", "what is", text)
        text = re.sub(r"\'ve ", " have ", text)
        text = re.sub(r"n\'t", " not ", text)
        text = re.sub(r"i\'m", "i am ", text)
        text = re.sub(r"I\'m", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r"c\+\+", "cplusplus", text)
        text = re.sub(r"c \+\+", "cplusplus", text)
        text = re.sub(r"c \+ \+", "cplusplus", text)
        text = re.sub(r"c#", "csharp", text)
        text = re.sub(r"f#", "fsharp", text)
        text = re.sub(r"g#", "gsharp", text)
        text = re.sub(r" e mail ", " email ", text)
        text = re.sub(r" e \- mail ", " email ", text)
        text = re.sub(r" e\-mail ", " email ", text)
        text = re.sub(r",000", '000', text)
        text = re.sub(r"\'s", " ", text)

        # spelling correction
        text = re.sub(r"ph\.d", "phd", text)
        text = re.sub(r"PhD", "phd", text)
        text = re.sub(r"pokemons", "pokemon", text)
        text = re.sub(r"pokémon", "pokemon", text)
        text = re.sub(r"pokemon go ", "pokemon-go ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" 9 11 ", " 911 ", text)
        text = re.sub(r" j k ", " jk ", text)
        text = re.sub(r" fb ", " facebook ", text)
        text = re.sub(r"facebooks", " facebook ", text)
        text = re.sub(r"facebooking", " facebook ", text)
        text = re.sub(r"insidefacebook", "inside facebook", text)
        text = re.sub(r"donald trump", "trump", text)
        text = re.sub(r"the big bang", "big-bang", text)
        text = re.sub(r"the european union", "eu", text)
        text = re.sub(r" usa ", " america ", text)
        text = re.sub(r" us ", " america ", text)
        text = re.sub(r" u s ", " america ", text)
        text = re.sub(r" U\.S\. ", " america ", text)
        text = re.sub(r" US ", " america ", text)
        text = re.sub(r" American ", " america ", text)
        text = re.sub(r" America ", " america ", text)
        text = re.sub(r" quaro ", " quora ", text)
        text = re.sub(r" mbp ", " macbook-pro ", text)
        text = re.sub(r" mac ", " macbook ", text)
        text = re.sub(r"macbook pro", "macbook-pro", text)
        text = re.sub(r"macbook-pros", "macbook-pro", text)
        text = re.sub(r" 1 ", " one ", text)
        text = re.sub(r" 2 ", " two ", text)
        text = re.sub(r" 3 ", " three ", text)
        text = re.sub(r" 4 ", " four ", text)
        text = re.sub(r" 5 ", " five ", text)
        text = re.sub(r" 6 ", " six ", text)
        text = re.sub(r" 7 ", " seven ", text)
        text = re.sub(r" 8 ", " eight ", text)
        text = re.sub(r" 9 ", " nine ", text)
        text = re.sub(r"googling", " google ", text)
        text = re.sub(r"googled", " google ", text)
        text = re.sub(r"googleable", " google ", text)
        text = re.sub(r"googles", " google ", text)
        text = re.sub(r" rs(\d+)", lambda m: ' rs ' + m.group(1), text)
        text = re.sub(r"(\d+)rs", lambda m: ' rs ' + m.group(1), text)
        text = re.sub(r"the european union", " eu ", text)
        text = re.sub(r"dollars", " dollar ", text)

        # punctuation
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"-", " - ", text)
        text = re.sub(r"/", " / ", text)
        text = re.sub(r"\\", " \ ", text)
        text = re.sub(r"=", " = ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r"\.", " . ", text)
        text = re.sub(r",", " , ", text)
        text = re.sub(r"\?", " ? ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\"", " \" ", text)
        text = re.sub(r"&", " & ", text)
        text = re.sub(r"\|", " | ", text)
        text = re.sub(r";", " ; ", text)
        text = re.sub(r"\(", " ( ", text)
        text = re.sub(r"\)", " ( ", text)

        # symbol replacement
        text = re.sub(r"&", " and ", text)
        text = re.sub(r"\|", " or ", text)
        text = re.sub(r"=", " equal ", text)
        text = re.sub(r"\+", " plus ", text)
        text = re.sub(r"₹", " rs ", text)      # 测试！
        text = re.sub(r"\$", " dollar ", text)

        # remove extra space
        text = ' '.join(text.split())

        return text

    @staticmethod
    def stem(df):
        """
        Process the text data with SnowballStemmer
        :param df: dataframe of original data
        :return: dataframe after stemming
        """
        df['question1'] = df.question1.map(lambda x: ' '.join(
            [TextPreProcessor._stemmer.stem(word) for word in
             nltk.word_tokenize(TextPreProcessor.clean_text(str(x).lower()).decode('utf-8'))]).encode('utf-8'))
        df['question2'] = df.question2.map(lambda x: ' '.join(
            [TextPreProcessor._stemmer.stem(word) for word in
             nltk.word_tokenize(TextPreProcessor.clean_text(str(x).lower()).decode('utf-8'))]).encode('utf-8'))
        return df


class DataPreprocessor(object):

    def __init__(self, config_fp):
        self.config = ConfigParser.ConfigParser()
        self.config.read(config_fp)

    @staticmethod
    def get_qid2question(df):
        """
        Get map(qid, question)
        :param df: original data set
        :return: map(qid, question)
        """
        qid2question = {}
        qids = df['qid1'].tolist() + df['qid2'].tolist()
        questions = df['question1'].tolist() + df['question2'].tolist()
        for ind in range(len(qids)):
            qid2question[qids[ind]] = questions[ind]
        LogUtil.log("INFO", "len(qids)=%d, len(unique_qids)=%d" % (len(qids), len(qid2question)))
        return qid2question

    @staticmethod
    def get_labels(df):
        """
        Get labels of data set
        :param df: original data set
        :return: label list of data set
        """
        labels = df['is_duplicate'].tolist()
        LogUtil.log("INFO", "num(1)=%d, num(0)=%d" % (sum(labels), len(labels) - sum(labels)))
        return labels

    @staticmethod
    def stat_dul_question(df):
        """
        Make statistics to duplication of questions
        :param df: original data set
        :return: none
        """
        questions = df['question1'].tolist() + df['question2'].tolist()
        len_questions = len(questions)
        len_uniq_questions = len(set(questions))
        LogUtil.log("INFO", "len(questions)=%d, len(unique_questions)=%d, rate=%f" % (
            len_questions, len_uniq_questions, 1.0 * len_uniq_questions / len_questions))

    @staticmethod
    def generate_cv_subset_index(cf, argv):
        """
        Generate index used for 5-fold cross validation
        :param cf: configuration file
        :param argv: parameter list
        :return: none
        """
        tag = argv[0]
        cv_num = 5
        cv_rawset_name = 'train_with_swap'
        train_data_size = 404290

        index_all = []
        for i in range(cv_num):
            index_all.append([])
        for i in range(train_data_size):
            index_all[int(random.random() * cv_num)].append(i)

        for i in range(cv_num):
            LogUtil.log('INFO', 'size(part%d)=%d' % (i, len(index_all[i])))

        index_fp = cf.get('DEFAULT', 'feature_index_pt')
        for i in range(cv_num):
            fold_id = i
            # train
            fp = '%s/cv_tag%s_n%d_f%d_train.%s.index' % (index_fp, tag, cv_num, fold_id, cv_rawset_name)
            for j in range(cv_num - 2):
                part_id = (i + j) % cv_num
                DataUtil.save_vector(fp, index_all[part_id], 'a')
            for j in range(cv_num - 2):
                part_id = (i + j) % cv_num
                DataUtil.save_vector(fp, [index + train_data_size for index in index_all[part_id]], 'a')
            # valid
            fp = '%s/cv_tag%s_n%d_f%d_valid.%s.index' % (index_fp, tag, cv_num, fold_id, cv_rawset_name)
            part_id = (fold_id + cv_num - 2) % cv_num
            DataUtil.save_vector(fp, index_all[part_id], 'w')
            # test
            fp = '%s/cv_tag%s_n%d_f%d_test.%s.index' % (index_fp, tag, cv_num, fold_id, cv_rawset_name)
            part_id = (fold_id + cv_num - 1) % cv_num
            DataUtil.save_vector(fp, index_all[part_id], 'w')

    @staticmethod
    def add_qid_for_test(df):
        """
        Add question ID for test data set
        :param df: original data set
        :return: dataframe of data set
        """
        df['qid1'] = df.apply(lambda r: ('T%08d' % (2 * r.test_id)), axis=1, raw=True)
        df['qid2'] = df.apply(lambda r: ('T%08d' % (2 * r.test_id + 1)), axis=1, raw=True)
        return df

    def swap_question(self):
        """
        Swap quesiton_1 and question_2 in original data set
        :return: none
        """
        train_swap_fp = '%s/train_swap.csv' % self.config.get('DEFAULT', 'devel_pt')
        # load `train.csv`
        train_data = pd.read_csv('%s/train.csv' % self.config.get('DEFAULT', 'origin_pt')).fillna(value="")  # [:100]
        # swap question
        offset = len(train_data)
        train_swap_data = train_data.apply(lambda x: [int(x.id) + offset,
                                                      x.qid2,
                                                      x.qid1,
                                                      x.question2,
                                                      x.question1,
                                                      x.is_duplicate], axis=1, raw=True)
        train_swap_data.to_csv(train_swap_fp, index=False)

    def generate_index_with_swap(self):
        """
        Generate the index file of `train_with_swap.csv`
        :return: none
        """
        train_index_fp = '%s/train_311.train.index' % self.config.get('DEFAULT', 'feature_index_pt')
        train_with_swap_index_fp = '%s/train_311.train_with_swap.index' % self.config.get('DEFAULT', 'feature_index_pt')

        train_index = DataUtil.load_vector(train_index_fp, False)
        train_index = [int(x) for x in train_index]

        offset = 404290
        train_swap_index = [x + offset for x in train_index]
        train_with_swap_index = train_index + train_swap_index
        DataUtil.save_vector(train_with_swap_index_fp, train_with_swap_index, 'w')