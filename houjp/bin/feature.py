# -*- coding: utf-8 -*-
#! /usr/bin/python

import ConfigParser
import sys
from scipy.sparse import csr_matrix, hstack

from utils import LogUtil

reload(sys)
sys.setdefaultencoding('utf-8')

class Feature(object):
	'''
	特征工程工具
	'''
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
	def demo():
		# 读取配置文件
		cf = ConfigParser.ConfigParser()
		cf.read("../conf/python.conf")
		# 加载特征文件
		features = Feature.load("%s/feature1.test.smat" % cf.get('path', 'feature_question_pt'))
		# 存储特征文件
		Feature.save(features, "%s/feature2.test.smat" % cf.get('path', 'feature_question_pt'))
		# 合并特征
		Feature.merge(features, features)
		# 获取<问题>特征池中的特征名
		Feature.get_feature_names_question(cf)


if __name__ == "__main__":
	Feature.demo()

