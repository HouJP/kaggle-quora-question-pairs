# -*- coding: utf-8 -*-
# ! /usr/bin/python

import pandas as pd
import ConfigParser
from utils import DataUtil
from utils import LogUtil
import nltk
from nltk.stem import SnowballStemmer
import re


class Preprocessor(object):
    '''
    预处理工具
    '''

    stemmer = SnowballStemmer('english')

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

    @staticmethod
    def clean_text(text):
        # 单位处理
        text = re.sub(r"(\d+)kgs ", lambda m: m.group(1) + ' kg ', text)        # e.g. 4kgs => 4 kg
        text = re.sub(r"(\d+)kg ", lambda m: m.group(1) + ' kg ', text)         # e.g. 4kg => 4 kg
        text = re.sub(r"(\d+)k ", lambda m: m.group(1) + '000 ', text)          # e.g. 4k => 4000
        text = re.sub(r"\$(\d+)", lambda m: m.group(1) + ' dollar ', text)
        text = re.sub(r"(\d+)\$", lambda m: m.group(1) + ' dollar ', text)

        # 缩略词处理
        text = re.sub(r"can\'t", "can not", text)
        text = re.sub(r"cannot", "can not ", text)
        text = re.sub(r"what\'s", "what is", text)
        text = re.sub(r"\'ve ", " have ", text)
        text = re.sub(r"n\'t", " not ", text)
        text = re.sub(r"i\'m", "i am ", text)
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

        # 拼写矫正及词条化
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

        # 标点符号处理
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"-", " - ", text)
        text = re.sub(r"/", " / ", text)
        text = re.sub(r"\\", " \ ", text)
        text = re.sub(r"=", " = ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r"\.", " . ", text)     # 单词与 . 之间插入空格
        text = re.sub(r",", " , ", text)      # 单词与 , 之间插入空格
        text = re.sub(r"\?", " ? ", text)     # 单词与 ? 之间插入空格
        text = re.sub(r"!", " ! ", text)      # 单词与 ! 之间插入空格
        text = re.sub(r"\"", " \" ", text)      # " 左右插入空格
        text = re.sub(r"&", " & ", text)        # & 左右插入空格
        text = re.sub(r"\|", " | ", text)        # | 左右插入空格
        text = re.sub(r";", " ; ", text)        # ; 左右插入空格
        text = re.sub(r"\(", " ( ", text)       # ( 左右插入空格
        text = re.sub(r"\)", " ( ", text)       # ) 左右插入空格

        # 符号替换为单词
        text = re.sub(r"&", " and ", text)
        text = re.sub(r"\|", " or ", text)
        text = re.sub(r"=", " equal ", text)
        text = re.sub(r"\+", " plus ", text)
        text = re.sub(r"₹", " rs ", text)      # 测试！
        text = re.sub(r"\$", " dollar ", text)

        # 去除多余空格
        text = ' '.join(text.split())

        return text

    @staticmethod
    def to_stem(df):
        """
        切词同时进行词干还原
        :param df:
        :return:
        """
        df['question1'] = df.question1.map(lambda x: ' '.join(
            [Preprocessor.stemmer.stem(word) for word in
             nltk.word_tokenize(Preprocessor.clean_text(str(x).lower()).decode('utf-8'))]).encode('utf-8'))
        df['question2'] = df.question2.map(lambda x: ' '.join(
            [Preprocessor.stemmer.stem(word) for word in
             nltk.word_tokenize(Preprocessor.clean_text(str(x).lower()).decode('utf-8'))]).encode('utf-8'))
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
        test_df.to_csv('%s/test_with_qid.csv' % cf.get('DEFAULT', 'devel_pt'), index=False)
        LogUtil.log('INFO', 'save test dataframe with qid done')

    @staticmethod
    def run_get_stem():
        # 读取配置文件
        cf = ConfigParser.ConfigParser()
        cf.read("../conf/python.conf")

        # 加载train.csv文件
        train_data = pd.read_csv('%s/train.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")#[:100]
        # 加载test.csv文件
        test_data = pd.read_csv('%s/test_with_qid.csv' % cf.get('DEFAULT', 'devel_pt')).fillna(value="")#[:100]

        # 存储文件路径
        train_stem_fp = '%s/stem.train.csv' % cf.get('DEFAULT', 'devel_pt')
        test_stem_fp = '%s/stem.test_with_qid.csv' % cf.get('DEFAULT', 'devel_pt')

        train_stem_data = Preprocessor.to_stem(train_data)
        test_stem_data = Preprocessor.to_stem(test_data)

        train_stem_data.to_csv(train_stem_fp, index=False)
        test_stem_data.to_csv(test_stem_fp, index=False)

    # @staticmethod
    # def swap_question(row):
    #     return [row["id"], row["qid2"],row["qid1"],row["question2"],row["question1"],row["is_duplicate"]]

    @staticmethod
    def run_question_swap():
        # 读取配置文件
        cf = ConfigParser.ConfigParser()
        cf.read("../conf/python.conf")
        train_swap_fp = '%s/train_swap.csv' % cf.get('DEFAULT', 'devel_pt')

        # 加载train.csv文件
        train_data = pd.read_csv('%s/train.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")  # [:100]

        # 交换question
        offset = len(train_data)
        train_swap_data = train_data.apply(lambda x: [int(x.id) + offset,
                                                      x.qid2,
                                                      x.qid1,
                                                      x.question2,
                                                      x.question1,
                                                      x.is_duplicate], axis=1, raw=True)
        train_swap_data.to_csv(train_swap_fp, index=False)

    @staticmethod
    def run_gen_index_with_swap():
        """
        生成线下训练集索引文件，包含swap部分
        :return:
        """
        # 读取配置文件
        cf = ConfigParser.ConfigParser()
        cf.read("../conf/python.conf")

        train_index_fp = '%s/train_311.train.index' % cf.get('DEFAULT', 'feature_index_pt')
        train_with_swap_index_fp = '%s/train_311.train_with_swap.index' % cf.get('DEFAULT', 'feature_index_pt')

        train_index = DataUtil.load_vector(train_index_fp, False)
        train_index = [int(x) for x in train_index]

        offset = 404290
        train_swap_index = [x + offset for x in train_index]

        train_with_swap_index = train_index + train_swap_index

        DataUtil.save_vector(train_with_swap_index_fp, train_with_swap_index, 'w')




if __name__ == "__main__":
    # PreprocessorRunner.get_qid2question(cf)
    # PreprocessorRunner.static_dul_question(cf)
    # PreprocessorRunner.get_labels(cf)
    # PreprocessorRunner.get_test_ids(cf)
    # PreprocessorRunner.get_test_indexs(cf)
    # PreprocessorRunner.get_test_labels(cf)
    # PreprocessorRunner.add_qid_for_test(cf)
    # PreprocessorRunner.run_get_stem()
    # PreprocessorRunner.run_question_swap()
    # PreprocessorRunner.run_gen_index_with_swap()
    text = 'hello| ph.d what\'s i\'m you\'re i\'d c ++ c# e mail i\'s PhD U.S.  rs123 1-2 fefefe. 1.2 world'
    print Preprocessor.clean_text(text)