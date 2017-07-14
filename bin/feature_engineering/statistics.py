#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/6/20 14:28
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


import math

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

from bin.featwheel.utils import MISSING_VALUE_NUMERIC
from bin.featwheel.utils import NgramUtil, DistanceUtil, LogUtil, MathUtil
from ..featwheel.extractor import Extractor
from ..preprocessor import TextPreProcessor

stops = set(stopwords.words("english"))
snowball_stemmer = SnowballStemmer('english')


class Not(Extractor):
    def __init__(self, config_fp):
        Extractor.__init__(self, config_fp)
        self.snowball_stemmer = SnowballStemmer('english')

    def get_feature_num(self):
        return 0

    def extract_row(self, row):
        q1 = str(row['question1']).strip()
        q2 = str(row['question2']).strip()

        q1_words = [self.snowball_stemmer.stem(word).encode('utf-8') for word in
                    nltk.word_tokenize(TextPreProcessor.clean_text(q1.decode('utf-8')))]
        q2_words = [self.snowball_stemmer.stem(word).encode('utf-8') for word in
                    nltk.word_tokenize(TextPreProcessor.clean_text(q2.decode('utf-8')))]

        not_cnt1 = q1_words.count('not')
        not_cnt2 = q2_words.count('not')

        fs = list()
        fs.append(not_cnt1)
        fs.append(not_cnt2)
        if not_cnt1 > 0 and not_cnt2 > 0:
            fs.append(1.)
        else:
            fs.append(0.)
        if (not_cnt1 > 0) or (not_cnt2 > 0):
            fs.append(1.)
        else:
            fs.append(0.)
        if not_cnt2 <= 0 < not_cnt1 or not_cnt1 <= 0 < not_cnt2:
            fs.append(1.)
        else:
            fs.append(0.)

        return fs


class WordMatchShare(Extractor):

    def extract_row(self, row):
        q1words = {}
        q2words = {}
        for word in str(row['question1']).lower().split():
            if word not in stops:
                q1words[word] = q1words.get(word, 0) + 1
        for word in str(row['question2']).lower().split():
            if word not in stops:
                q2words[word] = q2words.get(word, 0) + 1
        n_shared_word_in_q1 = sum([q1words[w] for w in q1words if w in q2words])
        n_shared_word_in_q2 = sum([q2words[w] for w in q2words if w in q1words])
        n_tol = sum(q1words.values()) + sum(q2words.values())
        if 1e-6 > n_tol:
            return [0.]
        else:
            return [1.0 * (n_shared_word_in_q1 + n_shared_word_in_q2) / n_tol]

    def get_feature_num(self):
        return 1


class TFIDFWordMatchShare(Extractor):

    def __init__(self, config):
        Extractor.__init__(self, config)

        train_data = pd.read_csv('%s/train.csv' % config.get('DIRECTORY', 'source_pt')).fillna(value="")
        self.idf = TFIDFWordMatchShare.init_idf(train_data)

    @staticmethod
    def init_idf(data):
        idf = {}
        q_set = set()
        for index, row in data.iterrows():
            q1 = str(row['question1'])
            q2 = str(row['question2'])
            if q1 not in q_set:
                q_set.add(q1)
                words = q1.lower().split()
                for word in words:
                    idf[word] = idf.get(word, 0) + 1
            if q2 not in q_set:
                q_set.add(q2)
                words = q2.lower().split()
                for word in words:
                    idf[word] = idf.get(word, 0) + 1
        num_docs = len(data)
        for word in idf:
            idf[word] = math.log(num_docs / (idf[word] + 1.)) / math.log(2.)
        LogUtil.log("INFO", "idf calculation done, len(idf)=%d" % len(idf))
        return idf

    def extract_row(self, row):
        q1words = {}
        q2words = {}
        for word in str(row['question1']).lower().split():
            q1words[word] = q1words.get(word, 0) + 1
        for word in str(row['question2']).lower().split():
            q2words[word] = q2words.get(word, 0) + 1
        sum_shared_word_in_q1 = sum([q1words[w] * self.idf.get(w, 0) for w in q1words if w in q2words])
        sum_shared_word_in_q2 = sum([q2words[w] * self.idf.get(w, 0) for w in q2words if w in q1words])
        sum_tol = sum(q1words[w] * self.idf.get(w, 0) for w in q1words) + sum(
            q2words[w] * self.idf.get(w, 0) for w in q2words)
        if 1e-6 > sum_tol:
            return [0.]
        else:
            return [1.0 * (sum_shared_word_in_q1 + sum_shared_word_in_q2) / sum_tol]

    def get_feature_num(self):
        return 1


class Length(Extractor):
    def extract_row(self, row):
        q1 = str(row['question1'])
        q2 = str(row['question2'])

        fs = list()
        fs.append(len(q1))
        fs.append(len(q2))
        fs.append(len(q1.split()))
        fs.append(len(q2.split()))
        return fs

    def get_feature_num(self):
        return 4


class LengthDiff(Extractor):
    def extract_row(self, row):
        q1 = row['question1']
        q2 = row['question2']
        return [abs(len(q1) - len(q2))]

    def get_feature_num(self):
        return 1


class LengthDiffRate(Extractor):
    def extract_row(self, row):
        len_q1 = len(row['question1'])
        len_q2 = len(row['question2'])
        if max(len_q1, len_q2) < 1e-6:
            return [0.0]
        else:
            return [1.0 * min(len_q1, len_q2) / max(len_q1, len_q2)]

    def get_feature_num(self):
        return 1


class PowerfulWord(object):
    @staticmethod
    def load_powerful_word(fp):
        powful_word = []
        f = open(fp, 'r')
        for line in f:
            subs = line.split('\t')
            word = subs[0]
            stats = [float(num) for num in subs[1:]]
            powful_word.append((word, stats))
        f.close()
        return powful_word

    @staticmethod
    def generate_powerful_word(data, subset_indexs):
        """
        计算数据中词语的影响力，格式如下：
            词语 --> [0. 出现语句对数量，1. 出现语句对比例，2. 正确语句对比例，3. 单侧语句对比例，4. 单侧语句对正确比例，5. 双侧语句对比例，6. 双侧语句对正确比例]
        """
        words_power = {}
        train_subset_data = data.iloc[subset_indexs, :]
        for index, row in train_subset_data.iterrows():
            label = int(row['is_duplicate'])
            q1_words = str(row['question1']).lower().split()
            q2_words = str(row['question2']).lower().split()
            all_words = set(q1_words + q2_words)
            q1_words = set(q1_words)
            q2_words = set(q2_words)
            for word in all_words:
                if word not in words_power:
                    words_power[word] = [0. for i in range(7)]
                # 计算出现语句对数量
                words_power[word][0] += 1.
                words_power[word][1] += 1.

                if ((word in q1_words) and (word not in q2_words)) or ((word not in q1_words) and (word in q2_words)):
                    # 计算单侧语句数量
                    words_power[word][3] += 1.
                    if 0 == label:
                        # 计算正确语句对数量
                        words_power[word][2] += 1.
                        # 计算单侧语句正确比例
                        words_power[word][4] += 1.
                if (word in q1_words) and (word in q2_words):
                    # 计算双侧语句数量
                    words_power[word][5] += 1.
                    if 1 == label:
                        # 计算正确语句对数量
                        words_power[word][2] += 1.
                        # 计算双侧语句正确比例
                        words_power[word][6] += 1.
        for word in words_power:
            # 计算出现语句对比例
            words_power[word][1] /= len(subset_indexs)
            # 计算正确语句对比例
            words_power[word][2] /= words_power[word][0]
            # 计算单侧语句对正确比例
            if words_power[word][3] > 1e-6:
                words_power[word][4] /= words_power[word][3]
            # 计算单侧语句对比例
            words_power[word][3] /= words_power[word][0]
            # 计算双侧语句对正确比例
            if words_power[word][5] > 1e-6:
                words_power[word][6] /= words_power[word][5]
            # 计算双侧语句对比例
            words_power[word][5] /= words_power[word][0]
        sorted_words_power = sorted(words_power.iteritems(), key=lambda d: d[1][0], reverse=True)
        LogUtil.log("INFO", "power words calculation done, len(words_power)=%d" % len(sorted_words_power))
        return sorted_words_power

    @staticmethod
    def save_powerful_word(words_power, fp):
        f = open(fp, 'w')
        for ele in words_power:
            f.write("%s" % ele[0])
            for num in ele[1]:
                f.write("\t%.5f" % num)
            f.write("\n")
        f.close()


class PowerfulWordDoubleSide(Extractor):

    def __init__(self, config_fp, thresh_num=500, thresh_rate=0.9):
        Extractor.__init__(self, config_fp)

        powerful_word_fp = '%s/words_power.%s.txt' % (
            self.config.get('DIRECTORY', 'devel_fp'), self.config.get('MODEL', 'train_subset_name'))
        self.pword = PowerfulWord.load_powerful_word(powerful_word_fp)
        self.pword_dside = PowerfulWordDoubleSide.init_powerful_word_dside(self.pword, thresh_num, thresh_rate)

    @staticmethod
    def init_powerful_word_dside(pword, thresh_num, thresh_rate):
        pword_dside = []
        pword = filter(lambda x: x[1][0] * x[1][5] >= thresh_num, pword)
        pword_sort = sorted(pword, key=lambda d: d[1][6], reverse=True)
        pword_dside.extend(map(lambda x: x[0], filter(lambda x: x[1][6] >= thresh_rate, pword_sort)))
        LogUtil.log('INFO', 'Double side power words(%d): %s' % (len(pword_dside), str(pword_dside)))
        return pword_dside

    def extract_row(self, row):
        tags = []
        q1_words = str(row['question1']).lower().split()
        q2_words = str(row['question2']).lower().split()
        for word in self.pword_dside:
            if (word in q1_words) and (word in q2_words):
                tags.append(1.0)
            else:
                tags.append(0.0)
        return tags

    def get_feature_num(self):
        return len(self.pword_dside)


class PowerfulWordOneSide(Extractor):

    def __init__(self, config_fp, thresh_num=500, thresh_rate=0.9):
        Extractor.__init__(self, config_fp)

        powerful_word_fp = '%s/words_power.%s.txt' % (
            self.config.get('DIRECTORY', 'devel_fp'), self.config.get('MODEL', 'train_subset_name'))
        self.pword = PowerfulWord.load_powerful_word(powerful_word_fp)
        self.pword_oside = PowerfulWordOneSide.init_powerful_word_oside(self.pword, thresh_num, thresh_rate)

    @staticmethod
    def init_powerful_word_oside(pword, thresh_num, thresh_rate):
        pword_oside = []
        pword = filter(lambda x: x[1][0] * x[1][3] >= thresh_num, pword)
        pword_oside.extend(
            map(lambda x: x[0], filter(lambda x: x[1][4] >= thresh_rate, pword)))
        LogUtil.log('INFO', 'One side power words(%d): %s' % (
            len(pword_oside), str(pword_oside)))
        return pword_oside

    def extract_row(self, row):
        tags = []
        q1_words = set(str(row['question1']).lower().split())
        q2_words = set(str(row['question2']).lower().split())
        for word in self.pword_oside:
            if (word in q1_words) and (word not in q2_words):
                tags.append(1.0)
            elif (word not in q1_words) and (word in q2_words):
                tags.append(1.0)
            else:
                tags.append(0.0)
        return tags

    def get_feature_num(self):
        return len(self.pword_oside)


class PowerfulWordDoubleSideRate(Extractor):
    def __init__(self, config_fp):
        Extractor.__init__(self, config_fp)

        powerful_word_fp = '%s/words_power.%s.txt' % (
            self.config.get('DIRECTORY', 'devel_fp'), self.config.get('MODEL', 'train_subset_name'))
        self.pword_dict = dict(PowerfulWord.load_powerful_word(powerful_word_fp))

    def extract_row(self, row):
        num_least = 300
        rate = [1.0]
        q1_words = set(str(row['question1']).lower().split())
        q2_words = set(str(row['question2']).lower().split())
        share_words = list(q1_words.intersection(q2_words))
        for word in share_words:
            if word not in self.pword_dict:
                continue
            if self.pword_dict[word][0] * self.pword_dict[word][5] < num_least:
                continue
            rate[0] *= (1.0 - self.pword_dict[word][6])
        rate = [1 - num for num in rate]
        return rate

    def get_feature_num(self):
        return 1


class PowerfulWordOneSideRate(Extractor):
    def __init__(self, config_fp):
        Extractor.__init__(self, config_fp)

        powerful_word_fp = '%s/words_power.%s.txt' % (
            self.config.get('DIRECTORY', 'devel_fp'), self.config.get('MODEL', 'train_subset_name'))
        self.pword_dict = dict(PowerfulWord.load_powerful_word(powerful_word_fp))

    def extract_row(self, row):
        num_least = 300
        rate = [1.0]
        q1_words = set(str(row['question1']).lower().split())
        q2_words = set(str(row['question2']).lower().split())
        q1_diff = list(set(q1_words).difference(set(q2_words)))
        q2_diff = list(set(q2_words).difference(set(q1_words)))
        all_diff = set(q1_diff + q2_diff)
        for word in all_diff:
            if word not in self.pword_dict:
                continue
            if self.pword_dict[word][0] * self.pword_dict[word][3] < num_least:
                continue
            rate[0] *= (1.0 - self.pword_dict[word][4])
        rate = [1 - num for num in rate]
        return rate

    def get_feature_num(self):
        return 1


class TFIDF(Extractor):
    def __init__(self, config_fp):
        Extractor.__init__(self, config_fp)
        self.tfidf = self.init_tfidf()

    def init_tfidf(self):
        train_data = pd.read_csv('%s/train.csv' % self.config.get('DIRECTORY', 'origin_pt')).fillna(value="")  # [:100]
        test_data = pd.read_csv('%s/test.csv' % self.config.get('DIRECTORY', 'origin_pt')).fillna(value="")  # [:100]

        tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 1))
        tfidf_txt = pd.Series(
            train_data['question1'].tolist() + train_data['question2'].tolist() + test_data['question1'].tolist() +
            test_data['question2'].tolist()).astype(str)
        tfidf.fit_transform(tfidf_txt)
        LogUtil.log("INFO", "init tfidf done ")
        return tfidf

    def extract_row(self, row):
        q1 = str(row['question1'])
        q2 = str(row['question2'])

        fs = list()
        fs.append(np.sum(self.tfidf.transform([str(q1)]).data))
        fs.append(np.sum(self.tfidf.transform([str(q2)]).data))
        fs.append(np.mean(self.tfidf.transform([str(q1)]).data))
        fs.append(np.mean(self.tfidf.transform([str(q2)]).data))
        fs.append(len(self.tfidf.transform([str(q1)]).data))
        fs.append(len(self.tfidf.transform([str(q2)]).data))
        return fs

    def get_feature_num(self):
        return 6


class DulNum(Extractor):
    @staticmethod
    def generate_dul_num(config):
        # load data set
        train_data = pd.read_csv('%s/train.csv' % config.get('DIRECTORY', 'source_pt')).fillna(value="")
        test_data = pd.read_csv('%s/test_with_qid.csv' % config.get('DIRECTORY', 'source_pt')).fillna(value="")

        dul_num = {}
        for index, row in train_data.iterrows():
            q1 = str(row.question1).strip()
            q2 = str(row.question2).strip()
            dul_num[q1] = dul_num.get(q1, 0) + 1
            if q1 != q2:
                dul_num[q2] = dul_num.get(q2, 0) + 1
        for index, row in test_data.iterrows():
            q1 = str(row.question1).strip()
            q2 = str(row.question2).strip()
            dul_num[q1] = dul_num.get(q1, 0) + 1
            if q1 != q2:
                dul_num[q2] = dul_num.get(q2, 0) + 1
        return dul_num

    def __init__(self, config_fp):
        Extractor.__init__(self, config_fp)
        self.dul_num = DulNum.generate_dul_num(self.config)

    def extract_row(self, row):
        q1 = str(row['question1']).strip()
        q2 = str(row['question2']).strip()

        dn1 = self.dul_num[q1]
        dn2 = self.dul_num[q2]
        return [dn1, dn2, max(dn1, dn2), min(dn1, dn2)]

    def get_feature_num(self):
        return 4


class MathTag(Extractor):

    def extract_row(self, row):
        q1 = str(row['question1']).strip()
        q2 = str(row['question2']).strip()

        q1_cnt = q1.count('[math]')
        q2_cnt = q2.count('[math]')
        pair_and = int((0 < q1_cnt) and (0 < q2_cnt))
        pair_or = int((0 < q1_cnt) or (0 < q2_cnt))
        return [q1_cnt, q2_cnt, pair_and, pair_or]

    def get_feature_num(self):
        return 4


class EnCharCount(Extractor):

    def extract_row(self, row):
        s = 'abcdefghijklmnopqrstuvwxyz'

        q1 = str(row['question1']).strip().lower()
        q2 = str(row['question2']).strip().lower()
        fs1 = [0] * 26
        fs2 = [0] * 26
        for index in range(len(q1)):
            c = q1[index]
            if 0 <= s.find(c):
                fs1[s.find(c)] += 1
        for index in range(len(q2)):
            c = q2[index]
            if 0 <= s.find(c):
                fs2[s.find(c)] += 1
        return fs1 + fs2 + list(abs(np.array(fs1) - np.array(fs2)))

    def get_feature_num(self):
        return 26 * 3


class NgramJaccardCoef(Extractor):

    def extract_row(self, row):
        q1_words = [snowball_stemmer.stem(word).encode('utf-8') for word in
                    nltk.word_tokenize(TextPreProcessor.clean_text(str(row['question1']).decode('utf-8')))]
        q2_words = [snowball_stemmer.stem(word).encode('utf-8') for word in
                    nltk.word_tokenize(TextPreProcessor.clean_text(str(row['question2']).decode('utf-8')))]
        fs = list()
        for n in range(1, 4):
            q1_ngrams = NgramUtil.ngrams(q1_words, n)
            q2_ngrams = NgramUtil.ngrams(q2_words, n)
            fs.append(DistanceUtil.jaccard_coef(q1_ngrams, q2_ngrams))
        return fs

    def get_feature_num(self):
        return 4


class NgramDiceDistance(Extractor):

    def extract_row(self, row):
        q1_words = [snowball_stemmer.stem(word).encode('utf-8') for word in
                    nltk.word_tokenize(TextPreProcessor.clean_text(str(row['question1']).decode('utf-8')))]
        q2_words = [snowball_stemmer.stem(word).encode('utf-8') for word in
                    nltk.word_tokenize(TextPreProcessor.clean_text(str(row['question2']).decode('utf-8')))]
        fs = list()
        for n in range(1, 4):
            q1_ngrams = NgramUtil.ngrams(q1_words, n)
            q2_ngrams = NgramUtil.ngrams(q2_words, n)
            fs.append(DistanceUtil.dice_dist(q1_ngrams, q2_ngrams))
        return fs

    def get_feature_num(self):
        return 4


class Distance(Extractor):

    def __init__(self, config_fp, distance_mode):
        Extractor.__init__(self, config_fp)
        self.feature_name += '_%s' % distance_mode
        self.valid_distance_mode = ['edit_dist', 'compression_dist']
        assert distance_mode in self.valid_distance_mode, "Wrong aggregation_mode: %s" % distance_mode
        self.distance_mode = distance_mode
        self.distance_func = getattr(DistanceUtil, self.distance_mode)

    def extract_row(self, row):
        q1 = str(row['question1']).strip()
        q2 = str(row['question2']).strip()
        q1_stem = ' '.join([snowball_stemmer.stem(word).encode('utf-8') for word in
                            nltk.word_tokenize(TextPreProcessor.clean_text(str(row['question1']).decode('utf-8')))])
        q2_stem = ' '.join([snowball_stemmer.stem(word).encode('utf-8') for word in
                            nltk.word_tokenize(TextPreProcessor.clean_text(str(row['question2']).decode('utf-8')))])
        return [self.distance_func(q1, q2), self.distance_func(q1_stem, q2_stem)]

    def get_feature_num(self):
        return 2


class NgramDistance(Distance):

    def extract_row(self, row):
        q1_words = [snowball_stemmer.stem(word).encode('utf-8') for word in
                    nltk.word_tokenize(TextPreProcessor.clean_text(str(row['question1']).decode('utf-8')))]
        q2_words = [snowball_stemmer.stem(word).encode('utf-8') for word in
                    nltk.word_tokenize(TextPreProcessor.clean_text(str(row['question2']).decode('utf-8')))]

        fs = list()
        aggregation_modes_outer = ["mean", "max", "min", "median"]
        aggregation_modes_inner = ["mean", "std", "max", "min", "median"]
        for n_ngram in range(1, 4):
            q1_ngrams = NgramUtil.ngrams(q1_words, n_ngram)
            q2_ngrams = NgramUtil.ngrams(q2_words, n_ngram)

            val_list = list()
            for w1 in q1_ngrams:
                _val_list = list()
                for w2 in q2_ngrams:
                    s = self.distance_func(w1, w2)
                    _val_list.append(s)
                if len(_val_list) == 0:
                    _val_list = [MISSING_VALUE_NUMERIC]
                val_list.append(_val_list)
            if len(val_list) == 0:
                val_list = [[MISSING_VALUE_NUMERIC]]

            for mode_inner in aggregation_modes_inner:
                tmp = list()
                for l in val_list:
                    tmp.append(MathUtil.aggregate(l, mode_inner))
                fs.extend(MathUtil.aggregate(tmp, aggregation_modes_outer))
            return fs

    def get_feature_num(self):
        return 4 * 5


class InterrogativeWords(Extractor):
    uni_words = ['what', 'whi', 'which', 'how', 'where', 'when', 'if', 'can', 'should']
    do_words = ['doe', 'do', 'did']
    be_words = ['is', 'are']
    will_words = ['will', 'would']

    @staticmethod
    def count(words):
        counter = list()
        for word in InterrogativeWords.uni_words:
            counter.append(words[0:1].count(word))
        counter.append(0.)
        for word in InterrogativeWords.do_words:
            counter[len(counter) - 1] += words[0:1].count(word)
        counter.append(0.)
        for word in InterrogativeWords.be_words:
            counter[len(counter) - 1] += words[0:1].count(word)
        counter.append(0.)
        for word in InterrogativeWords.will_words:
            counter[len(counter) - 1] += words[0:1].count(word)
        return counter

    def extract_row(self, row):
        q1 = str(row['question1']).strip()
        q2 = str(row['question2']).strip()
        q1_words = [snowball_stemmer.stem(word).encode('utf-8') for word in
                    nltk.word_tokenize(TextPreProcessor.clean_text(q1.decode('utf-8')))]
        q2_words = [snowball_stemmer.stem(word).encode('utf-8') for word in
                    nltk.word_tokenize(TextPreProcessor.clean_text(q2.decode('utf-8')))]
        counter_1 = InterrogativeWords.count(q1_words)
        counter_2 = InterrogativeWords.count(q2_words)

        fs = list()
        for ind1 in range(len(counter_1)):
            for ind2 in range(ind1, len(counter_1)):
                if (counter_1[ind1] > 0 and counter_2[ind2] > 0) or (counter_1[ind2] > 0 and counter_2[ind1] > 0):
                    fs.append(1.)
                else:
                    fs.append(0.)
        return fs

    def get_feature_num(self):
        return (1 + 12) * 12 / 2


def demo():
    config_fp = '/Users/houjianpeng/Github/kaggle-quora-question-pairs/conf/featwheel.conf'

    Not(config_fp).extract('train')
    Not(config_fp).extract('test')


if __name__ == '__main__':
    demo()
