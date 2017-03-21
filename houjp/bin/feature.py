# -*- coding: utf-8 -*-
#! /usr/bin/python

import ConfigParser
import sys
from scipy.sparse import csr_matrix, hstack
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import math
import matplotlib.pyplot as plt
from utils import LogUtil, StrUtil
from sklearn.metrics import roc_auc_score
import random

from utils import DataUtil

reload(sys)
sys.setdefaultencoding('utf-8')

class Feature(object):
	'''
	特征工程工具
	'''

	# 停用词
	stops = set(stopwords.words("english"))
	# train.csv中IDF字典
	train_idf = {}

	def __init__(self):
		return

	@staticmethod
	def load(ft_pt):
		'''
		加载特征文件，特征文件格式如下：
		row_num col_num
		f1_index:f1_value f2_index:f2_value ...
		'''
		data = []
		indice = []
		indptr = [0]
		f = open(ft_pt)
		[row_num, col_num] = [int(num) for num in f.readline().strip().split()]
		for line in f:
			line = line.strip()
			subs = line.split()
			for sub in subs:
				[f_index, f_value] = sub.split(":")
				f_index = int(f_index)
				f_value = float(f_value)
				data.append(f_value)
				indice.append(f_index)
			indptr.append(len(data))
		f.close()
		LogUtil.log("INFO", "load feature file done (%s)" % ft_pt)
		return csr_matrix((data, indice, indptr), shape = (row_num, col_num))

	@staticmethod
	def save(features, ft_pt):
		'''
		存储特征文件
		'''
		(row_num, col_num) = features.shape
		data = features.data
		indice = features.indices
		indptr = features.indptr
		f = open(ft_pt, 'w')
		f.write("%d %d\n" % (row_num, col_num))
		ind_indptr = 1
		for ind_data in range(len(data)):
			f.write("%d:%f" % (indice[ind_data], data[ind_data]))
			if (ind_data < indptr[ind_indptr] - 1):
				f.write(" ")
			else:
				f.write("\n")
				ind_indptr += 1
		LogUtil.log("INFO", "save feature file done (%s)" % ft_pt)
		f.close()

	@staticmethod
	def save_dataframe(features, ft_pt):
		'''
		存储DataFrame特征文件
		'''
		f = open(ft_pt, 'w')
		f.write('%d %d\n' % (len(features), len(features[0])))
		for row in features:
			for ind in range(len(row)):
				f.write('%d:%f' % (ind, row[ind]))
				if (ind < len(row) - 1):
					f.write(' ')
				else:
					f.write('\n')
		f.close()
		LogUtil.log("INFO", "save dataframe feature done (%s)" % ft_pt)
		return

	@staticmethod
	def merge(features_1, features_2):
		'''
		横向合并特征矩阵，即为每个实例增加特征
		'''
		features = hstack([features_1, features_2])
		(row_num, col_num) = features.shape
		LogUtil.log("INFO", "merge feature done, shape=(%d,%d)" % (row_num, col_num))
		return features

	@staticmethod
	def get_feature_names_question(cf):
		'''
		获取针对<问题>的特征池中的特证名
		'''
		return cf.get('feature', 'feature_names_question').split()

	@staticmethod
	def get_feature_names_question_pair(cf):
		'''
		获取针对<问题，问题>二元组的特征池中的特征名
		'''
		return cf.get('feature', 'feature_names_question_pair').split()

	@staticmethod
	def sample_with_index(features, indexs):
		'''
		根据索引采样特征向量
		'''
		features_sampled = features[indexs, :]
		(row_num, col_num) = features_sampled.shape
		LogUtil.log("INFO", "sample feature done, shape=(%d,%d)" % (row_num, col_num))
		return features_sampled

	@staticmethod
	def load_index(fp):
		'''
		加载特征索引文件
		'''
		f = open(fp)
		indexs = [int(line) for line in f.readlines()]
		LogUtil.log("INFO", "load index done, len(index)=%d" % (len(indexs)))
		f.close()
		return indexs

	@staticmethod
	def balance_index(indexs, labels, rate):
		'''
		增加正样本或者负样本的比例，使得正样本的比例在rate附近
		'''
		pos_indexs = [ index for index in indexs if labels[index] == 1. ]
		neg_indexs = [ index for index in indexs if labels[index] == 0. ]
		origin_rate = 1.0 * len(pos_indexs) / len(indexs)
		LogUtil.log("INFO", "original: len(pos)=%d, len(neg)=%d, rate=%.2f%%" % (len(pos_indexs), len(neg_indexs), 100.0 * origin_rate))
		if (origin_rate < rate):
			# 始终采样负样本
			pos_indexs, neg_indexs = neg_indexs, pos_indexs
			origin_rate = 1.0 - origin_rate
			rate = 1.0 - rate
			LogUtil.log("INFO", "increase postive instances ...")
		else:
			LogUtil.log("INFO", "increase negtive instances ...")
		k = (1. - rate) * origin_rate / rate / (1 - origin_rate)
		LogUtil.log("INFO", "k=%.4f" % k)
		balance_indexs = pos_indexs
		while k > 1e-6:
			if (k > 1.):
				balance_indexs.extend(neg_indexs)
			else:
				balance_indexs.extend(random.sample(neg_indexs, int(k * len(neg_indexs))))
			k -= 1.
		pos_indexs = [ index for index in balance_indexs if labels[index] == 1. ]
		neg_indexs = [ index for index in balance_indexs if labels[index] == 0. ]
		balanced_rate = 1.0 * len(pos_indexs) / len(balance_indexs)
		LogUtil.log("INFO", "balanced: len(pos)=%d, len(neg)=%d, rate=%.2f%%" % (len(pos_indexs), len(neg_indexs), 100.0 * balanced_rate))
		return balance_indexs

	@staticmethod
	def cal_word_share_rate(row):
		'''
		计算共享词比例，不包括停用词
		'''
		q1words = {}
		q2words = {}
		for word in StrUtil.tokenize_doc_en(row['question1']):
			if word not in Feature.stops:
				q1words[word] = 1 if word not in q1words else (q1words[word] + 1)
		for word in StrUtil.tokenize_doc_en(row['question2']):
			if word not in Feature.stops:
				q2words[word] = 1 if word not in q2words else (q2words[word] + 1)
		if len(q1words) == 0 or len(q2words) == 0:
			# The computer-generated chaff includes a few questions that are nothing but stopwords
			return (0., 0., 0.)
		len_shared_words_in_q1 = sum([ q1words[w] for w in q1words.keys() if w in q2words ])
		len_shared_words_in_q2 = sum([ q2words[w] for w in q2words.keys() if w in q1words ])
		len_q1 = sum(q1words.values())
		len_q2 = sum(q2words.values())
		r_in_q1 = 1.0 * len_shared_words_in_q1 / len_q1
		r_in_q2 = 1.0 * len_shared_words_in_q2 / len_q2 
		r = 1.0 * (len_shared_words_in_q1 + len_shared_words_in_q2) / (len_q1 + len_q2)
		return (r_in_q1, r_in_q2, r)

	@staticmethod
	def extract_word_share_rate(df):
		'''
		抽取<Q1,Q2>特征：word_share
		'''
		features = df.apply(Feature.cal_word_share_rate, axis = 1, raw = True)
		LogUtil.log("INFO", "extract word_share feature done ")
		return features

	@staticmethod
	def plot_word_share_rate(word_share, train_data):
		'''
		绘制<Q1,Q2>特征：word_share
		'''
		word_share_all = word_share.apply(lambda x: x[2])
		LogUtil.log("INFO", 'Original AUC: %f' % roc_auc_score(train_data['is_duplicate'], word_share_all.fillna(0)))
		plt.figure(figsize=(15, 5))
		plt.hist(word_share_all[train_data['is_duplicate'] == 0].fillna(0), bins=20, normed=True, label='Not Duplicate')
		plt.hist(word_share_all[train_data['is_duplicate'] == 1].fillna(0), bins=20, normed=True, alpha=0.7, label='Duplicate')
		plt.legend()
		plt.title('Label distribution over word_match_share', fontsize=15)
		plt.xlabel('word_share', fontsize=15)
		plt.show()

	@staticmethod
	def get_idf(df):
		'''
		根据文档获取idf字典，包括停用词
		'''
		idf = {}
		for index, row in df.iterrows():
			words = set(StrUtil.tokenize_doc_en(row['question']))
			for word in words:
				idf[word] = 1 if word not in idf else (idf[word] + 1)
		num_documents = len(df)
		for word in idf:
			idf[word] = math.log(1.0 * num_documents / (idf[word] + 1.0)) / math.log(2.0) 
		LogUtil.log("INFO", "IDF calculation done, len(idf)=%d" % len(idf))
		return idf

	@staticmethod
	def save_idf(idf, out_fp):
		'''
		存储IDF文件
		'''
		f = open(out_fp, 'w')
		for word in idf:
			f.write("%s %f\n" % (word, idf[word]))
		LogUtil.log("INFO", "save IDF file done (%s)" % out_fp)
		f.close()

	@staticmethod
	def load_idf(in_fp):
		'''
		加载IDF文件
		'''
		idf = {}
		f = open(in_fp)
		for line in f:
			[word, value] = line.strip().split()
			idf[word] = float(value)
		f.close()
		LogUtil.log("INFO", "load IDF file done, len(idf)=%d" % len(idf))
		return idf

	@staticmethod
	def cal_word_share_tfidf_rate(row):
		'''
		根据tfidf计算共享词比例，包括停用词
		'''
		q1words = {}
		q2words = {}
		for word in StrUtil.tokenize_doc_en(row['question1']):
			q1words[word] = 1 if word not in q1words else (q1words[word] + 1)
		for word in StrUtil.tokenize_doc_en(row['question2']):
			q2words[word] = 1 if word not in q2words else (q2words[word] + 1)
		if len(q1words) == 0 or len(q2words) == 0:
			# The computer-generated chaff includes a few questions that are nothing but stopwords
			return (0., 0., 0.)
		tfidf_shared_words_in_q1 = sum([ q1words[w] * Feature.train_idf[w] for w in q1words.keys() if w in q2words ])
		tfidf_shared_words_in_q2 = sum([ q2words[w] * Feature.train_idf[w] for w in q2words.keys() if w in q1words ])
		tfidf_q1 = sum([  q1words[w] * Feature.train_idf[w] for w in q1words.keys() ])
		tfidf_q2 = sum([  q2words[w] * Feature.train_idf[w] for w in q2words.keys() ])
		r_in_q1 = 1.0 * tfidf_shared_words_in_q1 / tfidf_q1
		r_in_q2 = 1.0 * tfidf_shared_words_in_q2 / tfidf_q2 
		r = 1.0 * (tfidf_shared_words_in_q1 + tfidf_shared_words_in_q2) / (tfidf_q1 + tfidf_q2)
		return (r_in_q1, r_in_q2, r)

	@staticmethod
	def extract_word_share_tfidf_rate(df):
		'''
		抽取<Q1,Q2>特征：word_share_tfidf
		'''
		features = df.apply(Feature.cal_word_share_tfidf_rate, axis = 1, raw = True)
		LogUtil.log("INFO", "extract word_share_tfidf feature done " )
		return features

	@staticmethod
	def plot_word_share_tfidf_rate(word_share_tfidf, train_data):
		'''
		绘制<Q1,Q2>特征：word_share_tfidf
		'''
		word_share_tfidf_all = word_share_tfidf.apply(lambda x: x[2])
		LogUtil.log("INFO", 'TFIDF AUC: %f' % roc_auc_score(train_data['is_duplicate'], word_share_tfidf_all.fillna(0)))
		plt.figure(figsize=(15, 5))
		plt.hist(word_share_tfidf_all[train_data['is_duplicate'] == 0].fillna(0), bins=20, normed=True, label='Not Duplicate')
		plt.hist(word_share_tfidf_all[train_data['is_duplicate'] == 1].fillna(0), bins=20, normed=True, alpha=0.7, label='Duplicate')
		plt.legend()
		plt.title('Label distribution over tfidf_word_match_share', fontsize=15)
		plt.xlabel('word_share_tfidf', fontsize=15)
		plt.show()


	@staticmethod
	def demo():
		'''
		使用样例代码
		'''
		# 读取配置文件
		cf = ConfigParser.ConfigParser()
		cf.read("../conf/python.conf")

		# 加载特征文件
		features = Feature.load("%s/feature1.demo.smat" % cf.get('path', 'feature_question_pt'))
		# 存储特征文件
		Feature.save(features, "%s/feature2.demo.smat" % cf.get('path', 'feature_question_pt'))
		# 合并特征
		Feature.merge(features, features)
		# 获取<问题>特征池中的特征名
		Feature.get_feature_names_question(cf)
		# 加载索引文件
		indexs = Feature.load_index("%s/vali.demo.index" % cf.get('path', 'feature_index_pt'))
		# 根据索引对特征采样
		features = Feature.sample_with_index(features, indexs)

		# 加载train.csv文件
		train_data = pd.read_csv('%s/train.csv' % cf.get('path', 'origin_pt')).fillna(value="")

		# 抽取<Q1,Q2>特征：word_share
		out_fp = '%s/word_share.train.smat' % cf.get('path', 'feature_question_pair_pt')
		features = Feature.extract_word_share_rate(train_data)
		Feature.save_dataframe(features, out_fp)

		# 绘制<Q1,Q2>特征：word_share
		Feature.plot_word_share_rate(features, train_data)

		# 计算train.csv中的IDF
		train_qid2question_fp = '%s/train_qid2question.csv' % cf.get('path', 'devel_pt')
		train_qid2question = pd.read_csv(train_qid2question_fp).fillna(value="")
		Feature.train_idf = Feature.get_idf(train_qid2question)
		Feature.save_idf(Feature.train_idf, '%s/train.idf' % cf.get('path', 'devel_pt'))

		# 抽取<Q1,Q2>特征：word_share_tfidf
		Feature.train_idf = Feature.load_idf('%s/train.idf' % cf.get('path', 'devel_pt'))
		out_fp = '%s/word_share_tfidf.train.smat' % cf.get('path', 'feature_question_pair_pt')
		features = Feature.extract_word_share_tfidf_rate(train_data)
		Feature.save_dataframe(features, out_fp)

		# 绘制<Q1,Q2>特征：word_share_tfidf
		Feature.plot_word_share_tfidf_rate(features, train_data)

		# 正负样本均衡化
		rate = 0.165
		train311_train_indexs_fp = '%s/train_311.train.index' % cf.get('path', 'feature_index_pt')
		train311_train_indexs = Feature.load_index(train311_train_indexs_fp)
		train_labels_fp = '%s/train.label' % cf.get('path', 'feature_label_pt')
		train_labels = DataUtil.load_vector(train_labels_fp, True)
		balanced_indexs = Feature.balance_index(train311_train_indexs, train_labels, rate)

	@staticmethod
	def test():
		'''
		测试函数
		'''
		# 读取配置文件
		cf = ConfigParser.ConfigParser()
		cf.read("../conf/python.conf")

		# 正负样本均衡化
		rate = 0.165
		train311_train_indexs_fp = '%s/train_311.train.index' % cf.get('path', 'feature_index_pt')
		train311_train_indexs = Feature.load_index(train311_train_indexs_fp)
		train_labels_fp = '%s/train.label' % cf.get('path', 'feature_label_pt')
		train_labels = DataUtil.load_vector(train_labels_fp, True)
		balanced_indexs = Feature.balance_index(train311_train_indexs, train_labels, rate)


if __name__ == "__main__":
	Feature.test()

