# -*- coding: utf-8 -*-
#! /usr/bin/python

import ConfigParser
import sys
import xgboost as xgb
import scipy as sp
import math

from utils import LogUtil, DataUtil
from feature import Feature

reload(sys)
sys.setdefaultencoding('utf-8')

class Model(object):
	'''
	模型工具类
	'''
	def __init__(self):
		return

	@staticmethod
	def entropy_loss(labels, pred_fp):
		'''
		根据预测文件计算Entropy Loss
		'''
		epsilon = 1e-15
		score = [0.] *  len(labels)
		for line in open(pred_fp, 'r'):
			if "test_id" in line:
				continue
			idx, s = line.strip().split(',')
			s = float(s)
			s = max(epsilon, s)
			s = min(1 - epsilon, s)
			score[int(idx)] = s
		s = 0.
		for idx, l in enumerate(labels):
			assert l == 1 or l == 0
			s +=  - l * math.log(score[idx]) - (1. - l) * math.log( 1 - score[idx])
		s /= len(labels)
		LogUtil.log('INFO', 'Entropy loss : %f ...'%(s))

	@staticmethod
	def get_DMatrix(indexs, labels, features, rate):
		'''
		根据索引文件构造DMatrix
		'''
		# 正负样本均衡化
		indexs = Feature.balance_index(indexs, labels, rate)
		# 根据索引采样标签
		labels = [labels[index] for index in indexs]
		# 根据索引采样特征
		features = Feature.sample_with_index(features, indexs)
		# 构造DMatrix
		return xgb.DMatrix(features, label = labels)

	@staticmethod
	def save_pred(ids, preds, fp):
		'''
		存储预测结果
		'''
		f = open(fp, 'w')
		f.write('"test_id","is_duplicate"\n')
		assert len(ids) == len(preds), "len(ids)=%d, len(preds)=%d" % (len(ids), len(preds))
		for index in range(len(ids)):
			f.write('%s,%s\n' % (str(ids[index]), str(preds[index])))
		f.close()
		LogUtil.log('INFO', 'save prediction file done (%s)' % fp)
		pass

	@staticmethod
	def train_xgb():
		'''
		训练xgb模型
		'''
		# 读取配置文件
		cf = ConfigParser.ConfigParser()
		cf.read("../conf/python.conf")

		# 设置正样本比例
		pos_rate = float(cf.get('MODEL', 'pos_rate'))

		# 加载训练集索引文件
		train_indexs = Feature.load_index(cf.get('MODEL', 'train_indexs_fp'))
		# 加载训练集标签文件
		train_labels = DataUtil.load_vector(cf.get('MODEL', 'train_labels_fp'), True)
		# 加载特征文件
		train_features = Feature.load_all_features(cf, cf.get('MODEL', 'train_rawset_name'))
		# 获取训练集
		train_data = Model.get_DMatrix(train_indexs, train_labels, train_features, pos_rate)
		LogUtil.log("INFO", "training set generation done")

		# 加载验证集索引文件
		valid_indexs = Feature.load_index(cf.get('MODEL', 'valid_indexs_fp'))
		# 加载验证集标签文件
		valid_labels = train_labels
		# 加载验证集特征文件
		valid_features = train_features
		# 检查验证集与训练集是否由同一份数据文件生成
		if (cf.get('MODEL', 'valid_rawset_name') != cf.get('MODEL', 'train_rawset_name')):
			valid_labels = DataUtil.load_vector(cf.get('MODEL', 'valid_labels_fp'), True)
			valid_features = Feature.load_all_features(cf, cf.get('MODEL', 'valid_rawset_name'))
		# 获取验证集
		valid_data = Model.get_DMatrix(valid_indexs, valid_labels, valid_features, pos_rate)
		LogUtil.log("INFO", "validation set generation done")

		# 加载测试集索引文件
		test_indexs = Feature.load_index(cf.get('MODEL', 'test_indexs_fp'))
		# 加载验证集标签文件
		test_labels = train_labels
		# 加载验证集特征文件
		test_features = train_features
		# 设置测试集正样本比例
		test_pos_rate = pos_rate
		# 检查测试集与训练集是否由同一份数据文件生成
		if (cf.get('MODEL', 'test_rawset_name') != cf.get('MODEL', 'train_rawset_name')):
			test_labels = DataUtil.load_vector(cf.get('MODEL', 'test_labels_fp'), True)
			test_features = Feature.load_all_features(cf, cf.get('MODEL', 'test_rawset_name'))
			test_pos_rate = -1.0
		# 获取验证集
		test_data = Model.get_DMatrix(test_indexs, test_labels, test_features, test_pos_rate)
		LogUtil.log("INFO", "test set generation done")

		# 设置参数
		params = {}
		params['objective'] = 'binary:logistic'
		params['eval_metric'] = 'logloss'
		params['eta'] = 0.02
		params['max_depth'] = 4
		params['silent'] = 1
		watchlist = [(train_data, 'train'), (valid_data, 'valid')]

		# 训练模型
		model = xgb.train(params, train_data, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)

		# 进行预测
		pred_data = model.predict(test_data)

		# 加载测试集ID文件
		test_ids = range(test_data.num_row())
		if (cf.get('MODEL', 'test_rawset_name') != cf.get('MODEL', 'train_rawset_name')):
			test_ids = DataUtil.load_vector(cf.get('MODEL', 'test_ids_fp'), False)

		# 存储预测结果
		pred_data_fp = cf.get('MODEL', 'test_prediction_fp')
		Model.save_pred(test_ids, pred_data, pred_data_fp)

		# 评测线下测试集得分
		if (cf.get('MODEL', 'test_rawset_name') == cf.get('MODEL', 'train_rawset_name')):
			Model.entropy_loss(test_data.get_label(), pred_data_fp)

	@staticmethod
	def test():
		'''
		测试函数
		'''
		# XGBoost模型训练及预测
		Model.train_xgb()

if __name__ == "__main__":
	Model.test()
