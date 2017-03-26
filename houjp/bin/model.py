# -*- coding: utf-8 -*-
# ! /usr/bin/python

import ConfigParser
import sys
import xgboost as xgb
import pandas as pd
import math
import time
import os

from utils import LogUtil, DataUtil
from feature import Feature

class Model(object):
    """
    模型工具类
    """

    def __init__(self):
        return

    @staticmethod
    def entropy_loss(labels, pred_fp):
        '''
        根据预测文件计算Entropy Loss
        '''
        epsilon = 1e-15
        score = [0.] * len(labels)
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
            s += - l * math.log(score[idx]) - (1. - l) * math.log(1 - score[idx])
        s /= len(labels)
        LogUtil.log('INFO', 'Entropy loss : %f ...' % (s))
        return s

    @staticmethod
    def get_DMatrix(indexs, labels, features, rate):
        '''
        根据索引文件构造DMatrix
        '''
        # 正负样本均衡化
        balanced_indexs = Feature.balance_index(indexs, labels, rate)
        # 根据索引采样标签
        labels = [labels[index] for index in balanced_indexs]
        # 根据索引采样特征
        features = Feature.sample_with_index(features, balanced_indexs)
        # 构造DMatrix
        return (xgb.DMatrix(features, label=labels), balanced_indexs)

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
    def train_xgb(cf, tag=time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))):
        '''
        训练xgb模型
        '''
        # 新增配置
        cf.set('DEFAULT', 'tag', str(tag))

        # 创建输出目录
        out_pt = cf.get('DEFAULT', 'out_pt')
        out_pt_exists = os.path.exists(out_pt)
        if out_pt_exists:
            LogUtil.log("ERROR", 'out path (%s) already exists ' % out_pt)
            return
        else:
            os.mkdir(out_pt)
            os.mkdir(cf.get('DEFAULT', 'pred_pt'))
            os.mkdir(cf.get('DEFAULT', 'model_pt'))
            os.mkdir(cf.get('DEFAULT', 'fault_pt'))
            os.mkdir(cf.get('DEFAULT', 'conf_pt'))
            os.mkdir(cf.get('DEFAULT', 'score_pt'))
            LogUtil.log('INFO', 'out path (%s) created ' % out_pt)

        # 保存本次运行配置
        cf.write(open(cf.get('DEFAULT', 'conf_pt') + 'python.conf', 'w'))

        # 设置正样本比例
        pos_rate = float(cf.get('MODEL', 'pos_rate'))

        # 加载训练集索引文件
        train_indexs = Feature.load_index(cf.get('MODEL', 'train_indexs_fp'))
        # 加载训练集标签文件
        train_labels = DataUtil.load_vector(cf.get('MODEL', 'train_labels_fp'), True)
        # 加载特征文件
        train_features = Feature.load_all_features(cf, cf.get('MODEL', 'train_rawset_name'))
        # 获取训练集
        (train_data, train_balanced_indexs) = Model.get_DMatrix(train_indexs, train_labels, train_features, pos_rate)
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
        (valid_data, valid_balanced_indexs) = Model.get_DMatrix(valid_indexs, valid_labels, valid_features, pos_rate)
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
        if cf.get('MODEL', 'test_rawset_name') != cf.get('MODEL', 'train_rawset_name'):
            test_labels = DataUtil.load_vector(cf.get('MODEL', 'test_labels_fp'), True)
            test_features = Feature.load_all_features(cf, cf.get('MODEL', 'test_rawset_name'))
            test_pos_rate = -1.0
        # 获取验证集
        (test_data, test_balanced_indexs) = Model.get_DMatrix(test_indexs, test_labels, test_features, test_pos_rate)
        LogUtil.log("INFO", "test set generation done")

        # 设置参数
        params = {}
        params['objective'] = cf.get('XGBOOST_PARAMS', 'objective')
        params['eval_metric'] = cf.get('XGBOOST_PARAMS', 'eval_metric')
        params['eta'] = float(cf.get('XGBOOST_PARAMS', 'eta'))
        params['max_depth'] = cf.getint('XGBOOST_PARAMS', 'max_depth')
        params['subsample'] = float(cf.get('XGBOOST_PARAMS', 'subsample'))
        params['colsample_bytree'] = float(cf.get('XGBOOST_PARAMS', 'colsample_bytree'))
        params['min_child_weight'] = cf.getint('XGBOOST_PARAMS', 'min_child_weight')
        params['silent'] = cf.getint('XGBOOST_PARAMS', 'silent')
        params['num_round'] = cf.getint('XGBOOST_PARAMS', 'num_round')
        params['early_stop'] = cf.getint('XGBOOST_PARAMS', 'early_stop')
        params['nthread'] = cf.getint('XGBOOST_PARAMS', 'nthread')
        watchlist = [(train_data, 'train'), (valid_data, 'valid')]

        # 训练模型
        model = xgb.train(params,
                          train_data, params['num_round'],
                          watchlist,
                          early_stopping_rounds=params['early_stop'],
                          verbose_eval=10)

        # 打印参数
        LogUtil.log("INFO", 'params=%s, best_ntree_limit=%d' % (str(params), model.best_ntree_limit))

        # 存储模型
        model_fp = cf.get('DEFAULT', 'model_pt') + '/xgboost.model'
        model.save_model(model_fp)

        # 进行预测
        pred_train_data = model.predict(train_data, ntree_limit=model.best_ntree_limit)
        pred_valid_data = model.predict(valid_data, ntree_limit=model.best_ntree_limit)
        pred_test_data = model.predict(test_data, ntree_limit=model.best_ntree_limit)

        # 加载训练集ID文件
        train_ids = range(train_data.num_row())
        # 存储训练集预测结果
        pred_train_fp = cf.get('MODEL', 'train_prediction_fp')
        Model.save_pred(train_ids, pred_train_data, pred_train_fp)
        # 评测线训练集得分
        LogUtil.log('INFO', 'Evaluate train data ====>')
        score_train = Model.entropy_loss(train_data.get_label(), pred_train_fp)

        # 加载验证集ID文件
        valid_ids = range(valid_data.num_row())
        if cf.get('MODEL', 'valid_rawset_name') != cf.get('MODEL', 'train_rawset_name'):
            valid_ids = DataUtil.load_vector(cf.get('MODEL', 'valid_ids_fp'), False)
        # 存储训练集预测结果
        pred_valid_fp = cf.get('MODEL', 'valid_prediction_fp')
        Model.save_pred(valid_ids, pred_valid_data, pred_valid_fp)
        # 评测线训练集得分
        LogUtil.log('INFO', 'Evaluate valid data ====>')
        score_valid = Model.entropy_loss(valid_data.get_label(), pred_valid_fp)

        # 加载测试集ID文件
        test_ids = range(test_data.num_row())
        if cf.get('MODEL', 'test_rawset_name') != cf.get('MODEL', 'train_rawset_name'):
            test_ids = DataUtil.load_vector(cf.get('MODEL', 'test_ids_fp'), False)
        # 存储测试集预测结果
        pred_test_fp = cf.get('MODEL', 'test_prediction_fp')
        Model.save_pred(test_ids, pred_test_data, pred_test_fp)
        # 评测线下测试集得分
        LogUtil.log('INFO', 'Evaluate test data ====>')
        score_test = Model.entropy_loss(test_data.get_label(), pred_test_fp)

        # 存储预测分数
        DataUtil.save_vector(cf.get('DEFAULT', 'score_pt') + 'score.txt',
                             ['score_train\t' + str(score_train),
                              'score_valid\t' + str(score_valid),
                              'score_test\t' + str(score_test)],
                             'w')

        # 存储预测不佳结果
        pos_fault_fp = cf.get('MODEL', 'pos_fault_fp')
        neg_fault_fp = cf.get('MODEL', 'neg_fault_fp')
        train_df = pd.read_csv(cf.get('MODEL', 'origin_pt') + '/train.csv')
        Model.generate_fault_file(pred_test_data, test_balanced_indexs, train_df, pos_fault_fp, neg_fault_fp)

        # 加载线上测试集索引文件
        online_test_indexs = Feature.load_index(cf.get('MODEL', 'online_test_indexs_fp'))
        # 加载线上测试集标签文件
        online_test_labels = DataUtil.load_vector(cf.get('MODEL', 'online_test_labels_fp'), True)
        # 加载线上测试集特征文件
        online_test_features = Feature.load_all_features(cf, cf.get('MODEL', 'online_test_rawset_name'))
        # 设置测试集正样本比例
        online_test_pos_rate = -1.0
        # 获取线上测试集
        (online_test_data, online_test_balanced_indexs) = Model.get_DMatrix(online_test_indexs, online_test_labels, online_test_features, online_test_pos_rate)
        LogUtil.log("INFO", "online test set generation done")

        # 预测线上测试集
        pred_online_test_data = model.predict(online_test_data, ntree_limit=model.best_ntree_limit)
        # 加载线上测试集ID文件
        online_test_ids = DataUtil.load_vector(cf.get('MODEL', 'online_test_ids_fp'), False)
        # 存储线上测试集预测结果
        pred_online_test_fp = cf.get('MODEL', 'online_test_prediction_fp')
        Model.save_pred(online_test_ids, pred_online_test_data, pred_online_test_fp)

    @staticmethod
    def generate_fault_file(pred_test_data, test_balanced_indexs, df, pos_fault_fp, neg_fault_fp):
        """
        生成预测成绩不佳的实例文件
        :param pred_test_data:
        :param test_balanced_indexs:
        :param df:
        :param pos_fault_fp:
        :param neg_fault_fp:
        :return:
        """
        pos = {}
        neg = {}
        for i in range(len(pred_test_data)):
            index = test_balanced_indexs[i]
            score = pred_test_data[i]
            label = df.loc[index]['is_duplicate']
            if (index in pos) or (index in neg):
                continue
            if 0 == label:
                neg[index] = (score, df.loc[index])
            else:
                pos[index] = (score, df.loc[index])
        pos = sorted(pos.iteritems(), key=lambda d: d[1][0], reverse=False)
        neg = sorted(neg.iteritems(), key=lambda d: d[1][0], reverse=True)

        f_pos = open(pos_fault_fp, 'w')
        for ele in pos:
            f_pos.write('%.5f\t%s\t||\t%s\t%d\t%d\n' % (ele[1][0], ele[1][1]['question1'], ele[1][1]['question2'], ele[1][1]['id'], ele[1][1]['is_duplicate']))
        f_pos.close()

        f_neg = open(neg_fault_fp, 'w')
        for ele in neg:
            f_neg.write('%.5f\t%s\t||\t%s\t%d\t%d\n' % (ele[1][0], ele[1][1]['question1'], ele[1][1]['question2'], ele[1][1]['id'], ele[1][1]['is_duplicate']))
        f_neg.close()

        LogUtil.log('INFO', 'save fault file done')
        LogUtil.log('INFO', 'pos_fault_fp=%s' % pos_fault_fp)
        LogUtil.log('INFO', 'neg_fault_fp=%s' % neg_fault_fp)

    @staticmethod
    def demo():
        """
        使用样例代码
        :return: NONE
        """
        # 读取配置文件
        cf = ConfigParser.ConfigParser()
        cf.read("../conf/python.conf")

        # XGBoost模型训练及预测
        Model.train_xgb(cf)


if __name__ == "__main__":
    Model.demo()
