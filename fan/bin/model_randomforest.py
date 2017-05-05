# -*- coding: utf-8 -*-
# ! /usr/bin/python

import ConfigParser
import sys
import pandas as pd
import math
import time
import os
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot
from utils import LogUtil, DataUtil
from feature import Feature
#from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import cPickle

class Model(object):
    """
    模型工具类
    """

    def __init__(self):
        return

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
    def get_SPMatrix(indexs, labels, features, rate):
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
        return features, labels, balanced_indexs

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
    def train_liblinear(cf, tag=time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))):
        '''
        训练liblinear模型
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
        (train_data, train_label, train_balanced_indexs) = Model.get_SPMatrix(train_indexs, train_labels, train_features, train_pos_rate)
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
        (valid_data, valid_label, valid_balanced_indexs) = Model.get_SPMatrix(valid_indexs, valid_labels, valid_features, valid_pos_rate)
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
        (test_data, test_label, test_balanced_indexs) = Model.get_SPMatrix(test_indexs, test_labels, test_features, test_pos_rate)
        LogUtil.log("INFO", "test set generation done")

        # 设置参数
        params = {}
        params['n_estimators'] = cf.getint('RANDOMFOREST_PARAMS', 'n_estimators') # number of trees, default=10
        params['criterion'] = cf.get('RANDOMFOREST_PARAMS', 'criterion').lower() # default='gini'
        params['max_features'] = float(cf.get('RANDOMFOREST_PARAMS', 'max_features'))
        params['max_depth'] = cf.getint('RANDOMFOREST_PARAMS', 'max_depth')
        params['min_samples_split'] = cf.getint('RANDOMFOREST_PARAMS', 'min_samples_split')
        params['min_samples_leaf'] = cf.getint('RANDOMFOREST_PARAMS', 'min_samples_leaf')
        params['min_weight_fraction_leaf'] = cf.getint('RANDOMFOREST_PARAMS', 'min_weight_fraction_leaf')
        params['max_leaf_nodes'] = None #cf.getint('RANDOMFOREST_PARAMS', 'max_leaf_nodes')
        params['min_impurity_split'] = float(cf.get('RANDOMFOREST_PARAMS', 'min_impurity_split'))
        params['bootstrap'] = cf.get('RANDOMFOREST_PARAMS', 'bootstrap').lower() == 'true'
        params['oob_score'] = cf.get('RANDOMFOREST_PARAMS', 'oob_score').lower() == 'true'
        params['n_jobs'] = cf.getint('RANDOMFOREST_PARAMS', 'n_jobs')
        params['random_state'] = cf.getint('RANDOMFOREST_PARAMS', 'random_state')
        params['verbose'] = cf.getint('RANDOMFOREST_PARAMS', 'verbose')
        params['warm_start'] = cf.get('RANDOMFOREST_PARAMS', 'warm_start').lower() == 'true'
        params['class_weight'] = None #{ 1: 0.5, 0: 0.5} #cf.getint('RANDOMFOREST_PARAMS', 'class_weight')

        model = RandomForestClassifier(n_estimators=params['n_estimators'], criterion=params['criterion'], max_features=params['max_features'], max_depth=params['max_depth'], min_samples_split=params['min_samples_split'], min_samples_leaf=params['min_samples_leaf'], min_weight_fraction_leaf=params['min_weight_fraction_leaf'], max_leaf_nodes=params['max_leaf_nodes'], min_impurity_split=params['min_impurity_split'], bootstrap=params['bootstrap'], oob_score=params['oob_score'], n_jobs=params['n_jobs'], random_state=params['random_state'], verbose=params['verbose'], warm_start=params['warm_start'], class_weight=params['class_weight'] )

        # 训练模型
        model.fit(train_data, train_label)

        # 打印参数
        LogUtil.log("INFO", 'params=%s' % (str(params)))

        # 存储模型
        model_param = model.get_params()
        model_fp = cf.get('DEFAULT', 'model_pt') + '/randomforest.model'
        cPickle.dump(model_param, open(model_fp, 'w'))

        # 保存本次运行配置
        cf.write(open(cf.get('DEFAULT', 'conf_pt') + 'python.conf', 'w'))

        # 进行预测
        print model.predict_proba(train_data)[:10]
        pred_train_data = model.predict_proba(train_data)[:,1]
        pred_valid_data = model.predict_proba(valid_data)[:,1]
        pred_test_data = model.predict_proba(test_data)[:,1]

        # 后处理
        pred_train_data = [Model.adj(x) for x in pred_train_data]
        pred_valid_data = [Model.adj(x) for x in pred_valid_data]
        pred_test_data = [Model.adj(x) for x in pred_test_data]

        # 加载训练集ID文件
        train_ids = range(train_data.shape[0])
        # 存储训练集预测结果
        pred_train_fp = cf.get('MODEL', 'train_prediction_fp')
        Model.save_pred(train_ids, pred_train_data, pred_train_fp)
        # 评测线训练集得分
        LogUtil.log('INFO', 'Evaluate train data ====>')
        score_train = Model.entropy_loss(train_label, pred_train_fp)

        # 加载验证集ID文件
        valid_ids = range(valid_data.shape[0])
        if cf.get('MODEL', 'valid_rawset_name') != cf.get('MODEL', 'train_rawset_name'):
            valid_ids = DataUtil.load_vector(cf.get('MODEL', 'valid_ids_fp'), False)
        # 存储训练集预测结果
        pred_valid_fp = cf.get('MODEL', 'valid_prediction_fp')
        Model.save_pred(valid_ids, pred_valid_data, pred_valid_fp)
        # 评测线训练集得分
        LogUtil.log('INFO', 'Evaluate valid data ====>')
        score_valid = Model.entropy_loss(valid_label, pred_valid_fp)

        # 加载测试集ID文件
        test_ids = range(test_data.shape[0])
        if cf.get('MODEL', 'test_rawset_name') != cf.get('MODEL', 'train_rawset_name'):
            test_ids = DataUtil.load_vector(cf.get('MODEL', 'test_ids_fp'), False)
        # 存储测试集预测结果
        pred_test_fp = cf.get('MODEL', 'test_prediction_fp')
        Model.save_pred(test_ids, pred_test_data, pred_test_fp)
        # 评测线下测试集得分
        LogUtil.log('INFO', 'Evaluate test data ====>')
        score_test = Model.entropy_loss(test_label, pred_test_fp)

        # 存储预测分数
        DataUtil.save_vector(cf.get('DEFAULT', 'score_pt') + 'score.txt',
                             ['score_train\t' + str(score_train),
                              'score_valid\t' + str(score_valid),
                              'score_test\t' + str(score_test)],
                             'w')

        # 存储预测不佳结果
        #pos_fault_fp = cf.get('MODEL', 'pos_fault_fp')
        #neg_fault_fp = cf.get('MODEL', 'neg_fault_fp')
        #train_df = pd.read_csv(cf.get('MODEL', 'origin_pt') + '/train.csv')
        #Model.generate_fault_file(pred_test_data, test_balanced_indexs, train_df, pos_fault_fp, neg_fault_fp)

        # 线上预测
        if 'True' == cf.get('MODEL', 'online'):
            Model.predict_liblinear(cf, model, params)
        return

    @staticmethod
    def load_model(cf):
        # 加载模型
        model_fp = cf.get('DEFAULT', 'model_pt') + '/randomforest.model'
        params = {}

        params['n_estimators'] = cf.getint('RANDOMFOREST_PARAMS', 'n_estimators') # number of trees, default=10
        params['criterion'] = cf.get('RANDOMFOREST_PARAMS', 'criterion').lower() # default='gini'
        params['max_features'] = float(cf.get('RANDOMFOREST_PARAMS', 'max_features'))
        params['max_depth'] = cf.getint('RANDOMFOREST_PARAMS', 'max_depth')
        params['min_samples_split'] = cf.getint('RANDOMFOREST_PARAMS', 'min_samples_split')
        params['min_samples_leaf'] = cf.getint('RANDOMFOREST_PARAMS', 'min_samples_leaf')
        params['min_weight_fraction_leaf'] = cf.getint('RANDOMFOREST_PARAMS', 'min_weight_fraction_leaf')
        params['max_leaf_nodes'] = None #cf.getint('RANDOMFOREST_PARAMS', 'max_leaf_nodes')
        params['min_impurity_split'] = float(cf.get('RANDOMFOREST_PARAMS', 'min_impurity_split'))
        params['bootstrap'] = cf.get('RANDOMFOREST_PARAMS', 'bootstrap').lower() == 'true'
        params['oob_score'] = cf.get('RANDOMFOREST_PARAMS', 'oob_score').lower() == 'true'
        params['n_jobs'] = cf.getint('RANDOMFOREST_PARAMS', 'n_jobs')
        params['random_state'] = cf.getint('RANDOMFOREST_PARAMS', 'random_state')
        params['verbose'] = cf.getint('RANDOMFOREST_PARAMS', 'verbose')
        params['warm_start'] = cf.get('RANDOMFOREST_PARAMS', 'warm_start').lower() == 'true'
        params['class_weight'] = None #{ 1: 0.5, 0: 0.5} #cf.getint('RANDOMFOREST_PARAMS', 'class_weight')

        model = RandomForestClassifier(n_estimators=params['n_estimators'], criterion=params['criterion'], max_features=params['max_features'], max_depth=params['max_depth'], min_samples_split=params['min_samples_split'], min_samples_leaf=params['min_samples_leaf'], min_weight_fraction_leaf=params['min_weight_fraction_leaf'], max_leaf_nodes=params['max_leaf_nodes'], min_impurity_split=params['min_impurity_split'], bootstrap=params['bootstrap'], oob_score=params['oob_score'], n_jobs=params['n_jobs'], random_state=params['random_state'], verbose=params['verbose'], warm_start=params['warm_start'], class_weight=params['class_weight'] )
        model.set_params(cPickle.load(open(model_fp)))

        return model, params

    @staticmethod
    def predict_liblinear(cf, model, params):
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
            (online_test_data, online_test_label, online_test_balanced_indexs) = Model.get_SPMatrix(range(0, online_test_features.shape[0]),
                                                                                [0] * online_test_features.shape[0],
                                                                                online_test_features,
                                                                                online_test_pos_rate)
            LogUtil.log("INFO", "online test set (%02d) generation done" % id_part)

            # 预测线上测试集
            pred_online_test_data = model.predict(online_test_data)
            all_pred_online_test_data.extend(pred_online_test_data)
            LogUtil.log('INFO', 'online test set (%02d) predict done' % id_part)
        # 后处理
        all_pred_online_test_data = [Model.adj(x) for x in all_pred_online_test_data]

        # 加载线上测试集ID文件
        online_test_ids = DataUtil.load_vector(cf.get('MODEL', 'online_test_ids_fp'), False)
        # 存储线上测试集预测结果
        pred_online_test_fp = cf.get('MODEL', 'online_test_prediction_fp')
        Model.save_pred(online_test_ids, all_pred_online_test_data, pred_online_test_fp)

    @staticmethod
    def run_predict_liblinear(conf_file_path):
        """
        使liblinear进行模型预测
        :param tag:
        :return:
        """
        cf_old = ConfigParser.ConfigParser()
        cf_old.read(conf_file_path)

        # 加载模型
        model, params = Model.load_model(cf_old)

        # 进行预测
        Model.predict_liblinear(cf_old, model, params)




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
        Model.train_liblinear(cf)


def print_help():
    print 'model <conf_file_path> -->'
    print '\ttrain'

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print_help()
        exit(1)

    # 读取配置文件
    cf = ConfigParser.ConfigParser()
    cf.read(sys.argv[1])

    cmd = sys.argv[2]
    if 'train' == cmd:
        Model.train_liblinear(cf)
    else:
        print_help()


