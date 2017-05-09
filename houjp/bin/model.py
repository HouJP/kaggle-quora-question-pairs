# -*- coding: utf-8 -*-
# ! /usr/bin/python

import ConfigParser
import sys
import xgboost as xgb
import pandas as pd
import math
import time
import os
from xgboost import plot_importance
from matplotlib import pyplot
from utils import LogUtil, DataUtil
from feature import Feature
from postprocessor import PostProcessor

class Model(object):
    """
    模型工具类
    """

    def __init__(self):
        return

    @staticmethod
    def entropy_loss_from_list(labels, preds):
        epsilon = 1e-15
        s = 0.
        for idx, l in enumerate(labels):
            assert l == 1 or l == 0
            score = preds[idx]
            score = max(epsilon, score)
            score = min(1 - epsilon, score)
            s += - l * math.log(score) - (1. - l) * math.log(1 - score)
        if len(labels) > 0:
            s /= len(labels)
        else:
            s = -1
        LogUtil.log('INFO', 'Entropy loss : %f' % (s))
        return s

    @staticmethod
    def load_preds(preds_fp):
        epsilon = 1e-15
        preds = []
        for line in open(preds_fp, 'r'):
            if "test_id" in line:
                continue
            idx, s = line.strip().split(',')
            s = float(s)
            s = max(epsilon, s)
            s = min(1 - epsilon, s)
            preds.append(s)
        return preds

    @staticmethod
    def cal_scores_with_clique_size(cf, tag, preds_fp):

        # 加载预测结果
        preds = Model.load_preds(preds_fp)
        if cf.get('MODEL', 'has_postprocess') == 'True':
            preds = [Model.inverse_adj(y) for y in preds]

        # 加载标签文件
        labels = DataUtil.load_vector(cf.get('MODEL', 'train_labels_fp'), True)

        # 加载索引文件
        indexs = Feature.load_index(cf.get('MODEL', '%s_indexs_fp' % tag))

        # 获取标签
        labels = [labels[index] for index in indexs]

        # 评分
        LogUtil.log('INFO', '-------------------')
        LogUtil.log('INFO', 'Evaluate as a whole (%s)' % tag)
        Model.entropy_loss_from_list(labels, preds)

        thresh = 3
        # 设置参数
        feature_name = 'graph_edge_max_clique_size'
        # 特征存储路径
        feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
        train_feature_fp = '%s/%s.%s.smat' % (feature_pt, feature_name, cf.get('MODEL', 'train_rawset_name'))
        train_features = Feature.load(train_feature_fp).toarray()
        # 测试集特征
        fs = [train_features[index] for index in indexs]

        LogUtil.log('INFO', '-------------------')
        LogUtil.log('INFO', 'Evaluate clique_size < 3 (%s)' % tag)
        labels_l = [labels[index] for index in range(len(labels)) if fs[index] < thresh]
        preds_l = [preds[index] for index in range(len(labels)) if fs[index] < thresh]
        Model.entropy_loss_from_list(labels_l, preds_l)
        LogUtil.log('INFO', 'rate_labels_l=%f, rate_preds_l=%f' % (
            1. * sum(labels_l) / len(labels_l), 1. * sum(preds_l) / len(preds_l)))

        LogUtil.log('INFO', '-------------------')
        LogUtil.log('INFO', 'Evaluate clique_size = 3 (%s)' % tag)
        labels_m = [labels[index] for index in range(len(labels)) if fs[index] == thresh]
        preds_m = [preds[index] for index in range(len(labels)) if fs[index] == thresh]
        Model.entropy_loss_from_list(labels_m, preds_m)
        LogUtil.log('INFO', 'rate_labels_m=%f, rate_preds_m=%f' % (
            1. * sum(labels_m) / len(labels_m), 1. * sum(preds_m) / len(preds_m)))

        LogUtil.log('INFO', '-------------------')
        LogUtil.log('INFO', 'Evaluate clique_size > 3 (%s)' % tag)
        labels_r = [labels[index] for index in range(len(labels)) if fs[index] > thresh]
        preds_r = [preds[index] for index in range(len(labels)) if fs[index] > thresh]
        Model.entropy_loss_from_list(labels_r, preds_r)
        LogUtil.log('INFO', 'rate_labels_r=%f, rate_preds_r=%f' % (
            1. * sum(labels_r) / len(labels_r), 1. * sum(preds_r) / len(preds_r)))

    @staticmethod
    def inverse_adj(y, te=0.173, tr=0.369):
        a = te / tr
        b = (1 - te) / (1 - tr)
        return b * y / (a + (b - a) * y)

    @staticmethod
    def adj(x, te=0.173, tr=0.369):
        a = te / tr
        b = (1 - te) / (1 - tr)
        return a * x / (a * x + b * (1 - x))

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
        features = Feature.sample_row(features, balanced_indexs)
        # 构造DMatrix
        return xgb.DMatrix(features, label=labels), balanced_indexs

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
    def train_xgb(cf, tag=time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))):
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

        # 设置正样本比例
        train_pos_rate = float(cf.get('MODEL', 'train_pos_rate'))
        # 加载训练集索引文件
        train_indexs = Feature.load_index(cf.get('MODEL', 'train_indexs_fp'))
        # 加载训练集标签文件
        train_labels = DataUtil.load_vector(cf.get('MODEL', 'train_labels_fp'), True)
        # 加载特征文件
        will_save = ('True' == cf.get('FEATURE', 'will_save'))
        train_features = Feature.load_all_features(cf, cf.get('MODEL', 'train_rawset_name'), will_save=will_save)
        # 获取训练集
        (train_data, train_balanced_indexs) = Model.get_DMatrix(train_indexs, train_labels, train_features, train_pos_rate)
        LogUtil.log("INFO", "training set generation done")

        # 设置正样本比例
        valid_pos_rate = float(cf.get('MODEL', 'valid_pos_rate'))
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
        (valid_data, valid_balanced_indexs) = Model.get_DMatrix(valid_indexs, valid_labels, valid_features, valid_pos_rate)
        LogUtil.log("INFO", "validation set generation done")

        # 设置正样本比例
        test_pos_rate = float(cf.get('MODEL', 'test_pos_rate'))
        # 加载测试集索引文件
        test_indexs = Feature.load_index(cf.get('MODEL', 'test_indexs_fp'))
        # 加载验证集标签文件
        test_labels = train_labels
        # 加载验证集特征文件
        test_features = train_features
        # 检查测试集与训练集是否由同一份数据文件生成
        if cf.get('MODEL', 'test_rawset_name') != cf.get('MODEL', 'train_rawset_name'):
            test_labels = DataUtil.load_vector(cf.get('MODEL', 'test_labels_fp'), True)
            test_features = Feature.load_all_features(cf, cf.get('MODEL', 'test_rawset_name'))
            test_pos_rate = -1.0
        # 获取测试集
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
        params['scale_pos_weight'] = float(cf.get('XGBOOST_PARAMS', 'scale_pos_weight'))
        watchlist = [(train_data, 'train'), (valid_data, 'valid')]

        # 训练模型
        model = xgb.train(params,
                          train_data, params['num_round'],
                          watchlist,
                          early_stopping_rounds=params['early_stop'],
                          verbose_eval=10)

        # 打印参数
        LogUtil.log("INFO", 'params=%s, best_ntree_limit=%d' % (str(params), model.best_ntree_limit))

        # 新增配置
        params['best_ntree_limit'] = model.best_ntree_limit
        cf.set('XGBOOST_PARAMS', 'best_ntree_limit', model.best_ntree_limit)

        # 存储模型
        model_fp = cf.get('DEFAULT', 'model_pt') + '/xgboost.model'
        model.save_model(model_fp)

        # 保存本次运行配置
        cf.write(open(cf.get('DEFAULT', 'conf_pt') + 'python.conf', 'w'))

        # 进行预测
        pred_train_data = model.predict(train_data, ntree_limit=model.best_ntree_limit)
        pred_valid_data = model.predict(valid_data, ntree_limit=model.best_ntree_limit)
        pred_test_data = model.predict(test_data, ntree_limit=model.best_ntree_limit)

        LogUtil.log('INFO', '-------------------')
        LogUtil.log('INFO', 'Evaluate as a whole')

        # 后处理
        if cf.get('MODEL', 'has_postprocess') == 'True':
            pred_train_data = [Model.adj(x) for x in pred_train_data]
            pred_valid_data = [Model.adj(x) for x in pred_valid_data]
            pred_test_data = [Model.adj(x) for x in pred_test_data]

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

        # 分块评分
        Model.cal_scores_with_clique_size(cf, 'train', pred_train_fp)
        Model.cal_scores_with_clique_size(cf, 'valid', pred_valid_fp)
        Model.cal_scores_with_clique_size(cf, 'test', pred_test_fp)

        # 线上预测
        if 'True' == cf.get('MODEL', 'online'):
            Model.predict_xgb(cf, model, params)
        return

    @staticmethod
    def cv_xgb(cf, tag=time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))):
        """
        xgb模型的交叉验证
        :param cf:
        :param tag:
        :return:
        """
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

        # 加载参数
        will_save = ('True' == cf.get('FEATURE', 'will_save'))
        offline_rawset_name = cf.get('MODEL', 'offline_rawset_name')
        cv_num = cf.getint('MODEL', 'cv_num')
        index_fp = cf.get('DEFAULT', 'feature_index_pt')
        label_fp = cf.get('DEFAULT', 'feature_label_pt')

        # 加载特征文件
        offline_features = Feature.load_all_features(cf, offline_rawset_name, will_save=will_save)
        # 加载标签文件
        offline_labels = DataUtil.load_vector('%s/%s.label' % (label_fp, offline_rawset_name), True)

        offline_valid_pred_all = []
        offline_valid_label_all = []

        offline_test_pred_all = []
        offline_test_label_all = []
        offline_test_index_all = []

        params_all = []
        model_all = []

        # 交叉验证
        for fold_id in range(cv_num):
            LogUtil.log('INFO', 'cross validation, fold_id=%d begin' % fold_id)

            # 加载训练集索引
            offline_train_pos_rate = float(cf.get('MODEL', 'train_pos_rate'))
            offline_train_indexs_fp = '%s/cv_n%d_f%d_train.%s.index' % (index_fp, cv_num, fold_id, offline_rawset_name)
            offline_train_indexs = Feature.load_index(offline_train_indexs_fp)
            # 获取训练集
            (offline_train_data, offline_train_balanced_indexs) = Model.get_DMatrix(
                offline_train_indexs,
                offline_labels,
                offline_features,
                offline_train_pos_rate)
            LogUtil.log('INFO', 'offline train data generation done')

            # 加载验证集索引
            offline_valid_pos_rate = float(cf.get('MODEL', 'valid_pos_rate'))
            offline_valid_indexs_fp = '%s/cv_n%d_f%d_valid.%s.index' % (index_fp, cv_num, fold_id, offline_rawset_name)
            offline_valid_indexs = Feature.load_index(offline_valid_indexs_fp)
            # 获取训练集
            (offline_valid_data, offline_valid_balanced_indexs) = Model.get_DMatrix(
                offline_valid_indexs,
                offline_labels,
                offline_features,
                offline_valid_pos_rate)
            LogUtil.log('INFO', 'offline valid data generation done')

            # 加载测试集索引
            offline_test_pos_rate = float(cf.get('MODEL', 'test_pos_rate'))
            offline_test_indexs_fp = '%s/cv_n%d_f%d_test.%s.index' % (index_fp, cv_num, fold_id, offline_rawset_name)
            offline_test_indexs = Feature.load_index(offline_test_indexs_fp)
            # 获取训练集
            (offline_test_data, offline_test_balanced_indexs) = Model.get_DMatrix(
                offline_test_indexs,
                offline_labels,
                offline_features,
                offline_test_pos_rate)
            LogUtil.log('INFO', 'offline test data generation done')

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
            params['scale_pos_weight'] = float(cf.get('XGBOOST_PARAMS', 'scale_pos_weight'))
            watchlist = [(offline_train_data, 'train'), (offline_valid_data, 'valid')]

            # 训练模型
            model = xgb.train(params,
                              offline_train_data, params['num_round'],
                              watchlist,
                              early_stopping_rounds=params['early_stop'],
                              verbose_eval=10)

            # 打印参数
            LogUtil.log("INFO", 'params=%s, best_ntree_limit=%d' % (str(params), model.best_ntree_limit))
            # 新增配置
            params['best_ntree_limit'] = model.best_ntree_limit
            cf.set('XGBOOST_PARAMS', 'best_ntree_limit', model.best_ntree_limit)

            params_all.append(params)

            # 存储模型
            model_fp = cf.get('DEFAULT', 'model_pt') + '/cv_n%d_f%d.xgboost.model' % (cv_num, fold_id)
            model.save_model(model_fp)

            model_all.append(model)

            # 进行预测
            offline_pred_train_data = model.predict(offline_train_data, ntree_limit=model.best_ntree_limit)
            offline_pred_valid_data = model.predict(offline_valid_data, ntree_limit=model.best_ntree_limit)
            offline_pred_test_data = model.predict(offline_test_data, ntree_limit=model.best_ntree_limit)

            # 后处理
            if cf.get('MODEL', 'has_postprocess') == 'True':
                offline_pred_train_data = [Model.adj(x) for x in offline_pred_train_data]
                offline_pred_valid_data = [Model.adj(x) for x in offline_pred_valid_data]
                offline_pred_test_data = [Model.adj(x) for x in offline_pred_test_data]

            offline_valid_score = Model.entropy_loss_from_list(offline_valid_data.get_label(), offline_pred_valid_data)
            offline_test_score = Model.entropy_loss_from_list(offline_test_data.get_label(), offline_pred_test_data)
            LogUtil.log('INFO', '-------------------')
            LogUtil.log('INFO', 'Evaluate for fold_id(%d): valid_score(%s), test_score(%s)' % (fold_id, offline_valid_score, offline_test_score))

            offline_valid_pred_all.extend(list(offline_pred_valid_data))
            offline_valid_label_all.extend(list(offline_valid_data.get_label()))

            offline_test_pred_all.extend(list(offline_pred_test_data))
            offline_test_label_all.extend(list(offline_test_data.get_label()))
            offline_test_index_all.extend(list(offline_test_balanced_indexs))

            LogUtil.log('INFO', 'cross validation, fold_id=%d done' % fold_id)

        # 保存本次运行配置
        cf.write(open(cf.get('DEFAULT', 'conf_pt') + 'python.conf', 'w'))

        # 存储预测结果
        offline_valid_pred_all_fp = '%s/cv_n%d_valid.%s.pred' % (cf.get('DEFAULT', 'pred_pt'), cv_num, offline_rawset_name)
        Model.save_pred(range(len(offline_valid_pred_all)), offline_valid_pred_all, offline_valid_pred_all_fp)
        offline_test_pred_all_fp = '%s/cv_n%d_test.%s.pred' % (cf.get('DEFAULT', 'pred_pt'), cv_num, offline_rawset_name)
        Model.save_pred(range(len(offline_test_pred_all)), offline_test_pred_all, offline_test_pred_all_fp)

        # 评测得分
        offline_valid_score_all = Model.entropy_loss(offline_valid_label_all, offline_valid_pred_all_fp)
        offline_test_score_all = Model.entropy_loss(offline_test_label_all, offline_test_pred_all_fp)
        LogUtil.log('INFO', '-------------------')
        LogUtil.log('INFO', 'Evaluate for all: valid_score_all(%s), test_score_all(%s)' % (
            offline_valid_score_all, offline_test_score_all))

        # 存储预测不佳结果
        pos_fault_fp = cf.get('MODEL', 'pos_fault_fp')
        neg_fault_fp = cf.get('MODEL', 'neg_fault_fp')
        train_df = pd.read_csv(cf.get('MODEL', 'origin_pt') + '/train.csv')
        Model.generate_fault_file(offline_test_pred_all, offline_test_index_all, train_df, pos_fault_fp, neg_fault_fp)

        # 线上预测
        if 'True' == cf.get('MODEL', 'online'):
            Model.cv_predict_xgb(cf, model_all, params_all)
        return

    @staticmethod
    def cv_predict_xgb(cf, model_all, params_all):
        # 加载配置
        n_part = cf.getint('MODEL', 'n_part')
        cv_num = cf.getint('MODEL', 'cv_num')

        # 全部预测结果
        online_pred_all = []
        for fold_id in range(cv_num):
            online_pred_all.append([])

        for part_id in range(n_part):
            # 加载线上测试集特征文件
            will_save = ('True' == cf.get('FEATURE', 'will_save'))
            online_features = Feature.load_all_features_with_part_id(cf,
                                                                     cf.get('MODEL', 'online_test_rawset_name'),
                                                                     part_id, will_save=will_save)
            # 设置测试集正样本比例
            online_test_pos_rate = -1.0
            # 获取线上测试集
            (online_data, online_balanced_indexs) = Model.get_DMatrix(range(0, online_features.shape[0]),
                                                                      [0] * online_features.shape[0],
                                                                      online_features,
                                                                      online_test_pos_rate)
            LogUtil.log("INFO", "online set (%02d) generation done" % part_id)

            for fold_id in range(cv_num):
                # 预测线上测试集
                online_pred = model_all[fold_id].predict(online_data, ntree_limit=params_all[fold_id]['best_ntree_limit'])
                online_pred_all[fold_id].extend(online_pred)
                LogUtil.log('INFO', 'online set part_id(%d), fold_id(%d) predict done' % (part_id, fold_id))

        # 后处理
        if cf.get('MODEL', 'has_postprocess') == 'True':
            for fold_id in range(cv_num):
                online_pred_all[fold_id] = [Model.adj(x) for x in online_pred_all[fold_id]]
        # 加载线上测试集ID文件
        online_ids = DataUtil.load_vector(cf.get('MODEL', 'online_test_ids_fp'), False)
        # 存储线上测试集预测结果
        online_pred_fp_list = []
        for fold_id in range(cv_num):
            online_pred_fp = '%s/cv_n%d_f%d_online.%s.pred' % (cf.get('DEFAULT', 'pred_pt'), cv_num, fold_id, cf.get('MODEL', 'online_test_rawset_name'))
            Model.save_pred(online_ids, online_pred_all[fold_id], online_pred_fp)
            online_pred_fp_list.append(online_pred_fp)

        # 模型融合
        online_pred_merge_fp = '%s/cv_n%d_online.%s.pred' % (cf.get('DEFAULT', 'pred_pt'), cv_num, cf.get('MODEL', 'online_test_rawset_name'))
        online_pred_list = []
        for online_pred_fp in online_pred_fp_list:
            online_pred = PostProcessor.read_result(online_pred_fp)
            online_pred_list.append(online_pred)
        online_pred_merge = PostProcessor.merge_logit(online_pred_list)
        PostProcessor.write_result(online_pred_merge_fp, online_pred_merge)
        LogUtil.log('INFO', 'cv merge done(%s)' % online_pred_merge_fp)


    @staticmethod
    def load_model(cf):
        # 加载模型
        model_fp = cf.get('DEFAULT', 'model_pt') + '/xgboost.model'
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
        params['best_ntree_limit'] = cf.getint('XGBOOST_PARAMS', 'best_ntree_limit')
        model = xgb.Booster(params)
        model.load_model(model_fp)

        return model, params

    @staticmethod
    def predict_xgb(cf, model, params):
        # 加载配置
        n_part = cf.getint('MODEL', 'n_part')

        # 全部预测结果
        all_pred_online_test_data = []

        for id_part in range(n_part):
            # 加载线上测试集特征文件
            will_save = ('True' == cf.get('FEATURE', 'will_save'))
            online_test_features = Feature.load_all_features_with_part_id(cf,
                                                                          cf.get('MODEL', 'online_test_rawset_name'),
                                                                          id_part, will_save=will_save)
            # 设置测试集正样本比例
            online_test_pos_rate = -1.0
            # 获取线上测试集
            (online_test_data, online_test_balanced_indexs) = Model.get_DMatrix(range(0, online_test_features.shape[0]),
                                                                                [0] * online_test_features.shape[0],
                                                                                online_test_features,
                                                                                online_test_pos_rate)
            LogUtil.log("INFO", "online test set (%02d) generation done" % id_part)

            # 预测线上测试集
            pred_online_test_data = model.predict(online_test_data, ntree_limit=params['best_ntree_limit'])
            all_pred_online_test_data.extend(pred_online_test_data)
            LogUtil.log('INFO', 'online test set (%02d) predict done' % id_part)
        # 后处理
        if cf.get('MODEL', 'has_postprocess') == 'True':
            all_pred_online_test_data = [Model.adj(x) for x in all_pred_online_test_data]

        # 加载线上测试集ID文件
        online_test_ids = DataUtil.load_vector(cf.get('MODEL', 'online_test_ids_fp'), False)
        # 存储线上测试集预测结果
        pred_online_test_fp = cf.get('MODEL', 'online_test_prediction_fp')
        Model.save_pred(online_test_ids, all_pred_online_test_data, pred_online_test_fp)

    @staticmethod
    def run_predict_xgb(conf_file_path):
        """
        使用xgb进行模型预测
        :param tag:
        :return:
        """
        cf_old = ConfigParser.ConfigParser()
        cf_old.read(conf_file_path)

        # 加载模型
        model, params = Model.load_model(cf_old)

        # 进行预测
        Model.predict_xgb(cf_old, model, params)

    @staticmethod
    def run_show_feature_xgb(cf, argv):
        # 加载参数
        max_num_features = int(argv[0])
        ylim_end = int(argv[1])
        # 加载模型
        model, params = Model.load_model(cf)

        # 输出重要性
        plot_importance(model, max_num_features=max_num_features, ylim=(0, ylim_end))
        pyplot.show()

    @staticmethod
    def run_select_feature_xgb(cf, argv):
        # 加载参数
        amx_num_features = int(argv[0])
        # 加载模型
        model, params = Model.load_model(cf)

        # 输出重要性



    @staticmethod
    def save_all_feature(cf):
        # 存储训练集特征文件
        Feature.load_all_features(cf, cf.get('MODEL', 'train_rawset_name'), True)
        # 存储预测集特征文件
        n_part = cf.getint('MODEL', 'n_part')
        for id_part in range(n_part):
            Feature.load_all_features_with_part_id(cf,
                                                   cf.get('MODEL', 'online_test_rawset_name'),
                                                   id_part, True)

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

def print_help():
    print 'model <conf_file_path> -->'
    print '\ttrain_xgb'
    print '\tsave_all_feature'
    print '\tshow_feature_xgb <max_num_features> <ylim_end>'
    print '\tcv_xgb'

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print_help()
        exit(1)

    # 读取配置文件
    cf = ConfigParser.ConfigParser()
    cf.read(sys.argv[1])

    cmd = sys.argv[2]
    if 'train_xgb' == cmd:
        Model.train_xgb(cf)
    elif 'save_all_feature' == cmd:
        Model.save_all_feature(cf)
    elif 'show_feature_xgb' == cmd:
        Model.run_show_feature_xgb(cf, sys.argv[3:])
    elif 'cv_xgb' == cmd:
        Model.cv_xgb(cf)
    else:
        print_help()


