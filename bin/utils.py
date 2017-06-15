#! /usr/bin/python
# -*- coding: utf-8 -*-
import re
import time
import random
import sys
from difflib import SequenceMatcher
from sklearn.metrics.pairwise import cosine_similarity
try:
    import lzma
    import Levenshtein
except:
    pass


MISSING_VALUE_NUMERIC = -1


class StrUtil(object):
    """
    Tool of String
    """

    def __init__(self):
        pass

    @staticmethod
    def tokenize_doc_en(doc):
        doc = doc.decode('utf-8')
        token_pattern = re.compile(r'\b\w\w+\b')
        lower_doc = doc.lower()
        tokenize_doc = token_pattern.findall(lower_doc)
        tokenize_doc = tuple(w for w in tokenize_doc)
        return tokenize_doc


class LogUtil(object):
    """
    Tool of Log
    """

    def __init__(self):
        pass

    @staticmethod
    def log(typ, msg):
        """
        Print message of log
        :param typ: type of log
        :param msg: message of log
        :return: none
        """
        print "[%s]\t[%s]\t%s" % (TimeUtil.t_now(), typ, str(msg))
        sys.stdout.flush()
        return


class TimeUtil(object):
    """
    Tool of Time
    """
    def __init__(self):
        return

    @staticmethod
    def t_now():
        """
        Get the current time, e.g. `2016-12-27 17:14:01`
        :return: string represented current time
        """
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))

    @staticmethod
    def t_now_YmdH():
        return time.strftime("%Y-%m-%d-%H", time.localtime(time.time()))


class DataUtil(object):
    """
    Tool of data process
    """
    def __init__(self):
        return

    @staticmethod
    def save_dic2csv(dic, header, out_fp):
        """
        Save dict instance to disk with CSV format
        :param dic: dict instance
        :param header: header of CSV file
        :param out_fp: output file path
        :return: none
        """
        fout = open(out_fp, 'w')
        fout.write('%s\n' % header)
        for k in dic:
            fout.write('"%s","%s"\n' % (k, dic[k].replace("\"", "\"\"")))
        fout.close()

    @staticmethod
    def random_split(instances, rates):
        """
        Random split data set with rates
        :param instances: data set
        :param rates: Proportions of each part of the data
        :return: list of subsets
        """
        LogUtil.log("INFO", "random split data(N=%d) into %d parts, with rates(%s) ..." % (
            len(instances), len(rates), str(rates)))
        slices = []
        pre_sum_rates = []
        sum_rates = 0.0
        for rate in rates:
            slices.append([])
            pre_sum_rates.append(sum_rates + rate)
            sum_rates += rate
        for instance in instances:
            randn = random.random()
            for i in range(0, len(pre_sum_rates)):
                if (randn < pre_sum_rates[i]):
                    slices[i].append(instance)
                    break
        n_slices = []
        for slic in slices:
            n_slices.append(len(slic))
        LogUtil.log("INFO", "random split data done, with number of instances(%s)." % (str(n_slices)))
        return slices

    @staticmethod
    def load_vector(file_path, is_float):
        """
        Load vector from disk
        :param file_path: vector file path
        :param is_float: convert elements to float type
        :return: a vector in List type
        """
        vector = []
        f = open(file_path)
        for line in f:
            value = int(line.strip()) if is_float else line.strip()
            vector.append(value)
        f.close()
        LogUtil.log("INFO", "load vector done. length=%d" % (len(vector)))
        return vector

    @staticmethod
    def save_vector(file_path, vector, mode):
        """
        Save vector on disk
        :param file_path: vector file path
        :param vector: a vector in List type
        :param mode: mode of writing file
        :return: none
        """
        file = open(file_path, mode)
        for value in vector:
            file.write(str(value) + "\n")
        file.close()
        return

    @staticmethod
    def load_matrix(file_path):
        """
        Load matrix from disk
        :param file_path: matrix file path
        :return: a matrix in 2-dim List type
        """
        matrix = []
        file = open(file_path)
        for line in file:
            vector = line.strip().split(',')
            vector = [float(vector[i]) for i in range(len(vector))]
            matrix.append(vector)
        file.close()
        LogUtil.log("INFO", "load matrix done. size=(%d,%d)" % (len(matrix), len(matrix[0])))
        return matrix

    @staticmethod
    def save_matrix(file_path, instances, mode):
        """
        Save matrix on disk
        :param file_path: matrix file path
        :param instances: a matrix in 2-dim List type
        :param mode: mode of writing file
        :return: none
        """
        file = open(file_path, mode)
        for instance in instances:
            file.write(','.join([str(instance[i]) for i in range(len(instance))]))
            file.write('\n')
        file.close()
        return


class MathUtil(object):
    """
    Tool of Math
    """

    @staticmethod
    def count_one_bits(x):
        """
        Calculate the number of bits which are 1
        :param x: number which will be calculated
        :return: number of bits in `x`
        """
        n = 0
        while x:
            n += 1 if (x & 0x01) else 0
            x >>= 1
        return n

    @staticmethod
    def int2binarystr(x):
        """
        Convert the number from decimal to binary
        :param x: decimal number
        :return: string represented binary format of `x`
        """
        s = ""
        while x:
            s += "1" if (x & 0x01) else "0"
            x >>= 1
        return s[::-1]

    @staticmethod
    def try_divide(x, y, val=0.0):
        """
        try to divide two numbers
        """
        if y != 0.0:
            val = float(x) / y
        return val


class DistanceUtil(object):
    """
    Tool of Distance
    """

    @staticmethod
    def edit_dist(str1, str2):
        try:
            # very fast
            # http://stackoverflow.com/questions/14260126/how-python-levenshtein-ratio-is-computed
            import Levenshtein
            d = Levenshtein.distance(str1, str2) / float(max(len(str1), len(str2)))
        except:
            # https://docs.python.org/2/library/difflib.html
            d = 1. - SequenceMatcher(lambda x: x == " ", str1, str2).ratio()
        return d

    @staticmethod
    def is_str_match(str1, str2, threshold=1.0):
        assert threshold >= 0.0 and threshold <= 1.0, "Wrong threshold."
        if float(threshold) == 1.0:
            return str1 == str2
        else:
            return (1. - DistanceUtil.edit_dist(str1, str2)) >= threshold

    @staticmethod
    def longest_match_size(str1, str2):
        sq = SequenceMatcher(lambda x: x == " ", str1, str2)
        match = sq.find_longest_match(0, len(str1), 0, len(str2))
        return match.size

    @staticmethod
    def longest_match_ratio(str1, str2):
        sq = SequenceMatcher(lambda x: x == " ", str1, str2)
        match = sq.find_longest_match(0, len(str1), 0, len(str2))
        return MathUtil.try_divide(match.size, min(len(str1), len(str2)))

    @staticmethod
    def longest_match_size(str1, str2):
        sq = SequenceMatcher(lambda x: x == " ", str1, str2)
        match = sq.find_longest_match(0, len(str1), 0, len(str2))
        return match.size

    @staticmethod
    def longest_match_ratio(str1, str2):
        sq = SequenceMatcher(lambda x: x == " ", str1, str2)
        match = sq.find_longest_match(0, len(str1), 0, len(str2))
        return MathUtil.try_divide(match.size, min(len(str1), len(str2)))

    @staticmethod
    def compression_dist(x, y, l_x=None, l_y=None):
        if x == y:
            return 0
        x_b = x.encode('utf-8')
        y_b = y.encode('utf-8')
        if l_x is None:
            l_x = len(lzma.compress(x_b))
            l_y = len(lzma.compress(y_b))
        l_xy = len(lzma.compress(x_b + y_b))
        l_yx = len(lzma.compress(y_b + x_b))
        dist = MathUtil.try_divide(min(l_xy, l_yx) - min(l_x, l_y), max(l_x, l_y))
        return dist

    @staticmethod
    def cosine_sim(vec1, vec2):
        try:
            s = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]
        except:
            try:
                s = cosine_similarity(vec1, vec2)[0][0]
            except:
                s = MISSING_VALUE_NUMERIC
        return s

    @staticmethod
    def jaccard_coef(A, B):
        if not isinstance(A, set):
            A = set(A)
        if not isinstance(B, set):
            B = set(B)
        return MathUtil.try_divide(float(len(A.intersection(B))), len(A.union(B)))

    @staticmethod
    def dice_dist(A, B):
        if not isinstance(A, set):
            A = set(A)
        if not isinstance(B, set):
            B = set(B)
        return MathUtil.try_divide(2. * float(len(A.intersection(B))), (len(A) + len(B)))
