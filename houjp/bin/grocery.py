# -*- coding: utf-8 -*-
# ! /usr/bin/python

'''
该脚本中记录的是临时使用的函数
'''

import ConfigParser
import sys
from feature import Feature
import pandas as pd
import math
from utils import LogUtil
from utils import DataUtil
from model import Model

reload(sys)
sys.setdefaultencoding('utf-8')


def to_csv(in_fp, out_fp, header):
    '''
    将空格分隔的文件格式转化为CSV格式
    '''
    fin = open(in_fp)
    fout = open(out_fp, 'w')
    fout.write("%s\n" % header)
    for line in fin:
        subs = line.split()
        fout.write("%s\n" % ",".join(subs))
    fin.close()
    fout.close()
    return


def to_csv_run(cf):
    devel_pt = cf.get('path', 'devel_pt')

    in_fp = "%s/relation.devel-311.train.txt" % devel_pt
    out_fp = "%s/relation.devel-311.train.csv" % devel_pt
    header = "\"is_duplicate\",\"qid1\",\"qid2\""
    to_csv(in_fp, out_fp, header)

    in_fp = "%s/relation.devel-311.valid.txt" % devel_pt
    out_fp = "%s/relation.devel-311.valid.csv" % devel_pt
    header = "\"is_duplicate\",\"qid1\",\"qid2\""
    to_csv(in_fp, out_fp, header)

    in_fp = "%s/relation.devel-311.test.txt" % devel_pt
    out_fp = "%s/relation.devel-311.test.csv" % devel_pt
    header = "\"is_duplicate\",\"qid1\",\"qid2\""
    to_csv(in_fp, out_fp, header)

    in_fp = "%s/relation.devel-811.train.txt" % devel_pt
    out_fp = "%s/relation.devel-811.train.csv" % devel_pt
    header = "\"is_duplicate\",\"qid1\",\"qid2\""
    to_csv(in_fp, out_fp, header)

    in_fp = "%s/relation.devel-811.valid.txt" % devel_pt
    out_fp = "%s/relation.devel-811.valid.csv" % devel_pt
    header = "\"is_duplicate\",\"qid1\",\"qid2\""
    to_csv(in_fp, out_fp, header)

    in_fp = "%s/relation.devel-811.test.txt" % devel_pt
    out_fp = "%s/relation.devel-811.test.csv" % devel_pt
    header = "\"is_duplicate\",\"qid1\",\"qid2\""
    to_csv(in_fp, out_fp, header)

    return


def to_feature_index(data_fp, sub_data_fp, sub_data_index_fp):
    '''
    将空格分割的文件格式转化为索引文件
    '''
    p = open(data_fp).readlines()
    p = [line.split() for line in p]
    p_set = {}
    for index in range(len(p)):
        ele = p[index]
        p_set[(ele[1], ele[2])] = index
    print 'len(p)=%d,len(p_set)=%d' % (len(p), len(p_set))

    sub_p = open(sub_data_fp).readlines()
    sub_p = [line.split() for line in sub_p]
    sub_p_indexs = []

    for ele in sub_p:
        sub_p_indexs.append(p_set[(ele[1], ele[2])])

    print 'len(sub_p)=%d,len(sub_p_indexs)=%d' % (len(sub_p), len(sub_p_indexs))

    f = open(sub_data_index_fp, 'w')
    for index in sub_p_indexs:
        f.write('%d\n' % index)
    f.close()


def to_feature_index_run(cf):
    data_fp = '%s/relation.train.txt' % cf.get('path', 'devel_pt')

    train_311_fp = '%s/relation.devel-311.train.txt' % cf.get('path', 'devel_pt')
    train_311_index_fp = '%s/train_311.train.index' % cf.get('path', 'feature_index_pt')
    to_feature_index(data_fp, train_311_fp, train_311_index_fp)

    valid_311_fp = '%s/relation.devel-311.valid.txt' % cf.get('path', 'devel_pt')
    valid_311_index_fp = '%s/valid_311.train.index' % cf.get('path', 'feature_index_pt')
    to_feature_index(data_fp, valid_311_fp, valid_311_index_fp)

    test_311_fp = '%s/relation.devel-311.test.txt' % cf.get('path', 'devel_pt')
    test_311_index_fp = '%s/test_311.train.index' % cf.get('path', 'feature_index_pt')
    to_feature_index(data_fp, test_311_fp, test_311_index_fp)

    train_811_fp = '%s/relation.devel-811.train.txt' % cf.get('path', 'devel_pt')
    train_811_index_fp = '%s/train_811.train.index' % cf.get('path', 'feature_index_pt')
    to_feature_index(data_fp, train_811_fp, train_811_index_fp)

    valid_811_fp = '%s/relation.devel-811.valid.txt' % cf.get('path', 'devel_pt')
    valid_811_index_fp = '%s/valid_811.train.index' % cf.get('path', 'feature_index_pt')
    to_feature_index(data_fp, valid_811_fp, valid_811_index_fp)

    test_811_fp = '%s/relation.devel-811.test.txt' % cf.get('path', 'devel_pt')
    test_811_index_fp = '%s/test_811.train.index' % cf.get('path', 'feature_index_pt')
    to_feature_index(data_fp, test_811_fp, test_811_index_fp)


def generate_answer(cf):
    # 设置参数
    feature_name = 'graph_edge_max_clique_size'

    # 特征存储路径
    feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
    train_feature_fp = '%s/%s.train.smat' % (feature_pt, feature_name)
    test_feature_fp = '%s/%s.test.smat' % (feature_pt, feature_name)

    test_features = Feature.load_smat(test_feature_fp).toarray()

    lr = 0.4
    rr = 0.7
    thresh = 3
    fout = open('/Users/houjianpeng/tmp/tmp_%.2f_%.2f.csv' % (lr, rr), 'w')
    fout.write("\"test_id\",\"is_duplicate\"\n")

    for index in range(len(test_features)):
        if test_features[index][0] > thresh:
            fout.write('%d,%f\n' % (index, rr))
        else:
            fout.write('%d,%f\n' % (index, lr))
    fout.close()


def cal_pos_rate(cf):
    # 加载数据文件
    train_data = pd.read_csv('%s/train.csv' % cf.get('DEFAULT', 'origin_pt')).fillna(value="")
    labels = train_data['is_duplicate']

    # 设置参数
    feature_name = 'graph_edge_max_clique_size'
    # 特征存储路径
    feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
    train_feature_fp = '%s/%s.train.smat' % (feature_pt, feature_name)
    train_features = Feature.load(train_feature_fp).toarray()

    thresh = 2

    len_l = 0
    len_r = 0
    len_l_pos = 0
    len_r_pos = 0
    for index in range(len(labels)):
        if train_features[index][0] > thresh:
            len_r += 1.
            if labels[index] == 1:
                len_r_pos += 1.
        else:
            len_l += 1.
            if labels[index] == 1:
                len_l_pos += 1.
    print 'len_l=%d, len_r=%d, len_l_pos=%d, len_r_pos=%d' % (len_l, len_r, len_l_pos, len_r_pos)
    print 'pos_rate_l=%f, pos_rate_r=%f' % (len_l_pos / len_l, len_r_pos / len_r)


def entropy_loss(labels, preds):
    epsilon = 1e-15
    s = 0.
    for idx, l in enumerate(labels):
        assert l == 1 or l == 0
        score = preds[idx]
        score = max(epsilon, score)
        score = min(1 - epsilon, score)
        s += - l * math.log(score) - (1. - l) * math.log(1 - score)
    s /= len(labels)
    LogUtil.log('INFO', 'Entropy loss : %f' % (s))
    return s


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


def cal_scores():
    test_preds_fp = '/Users/houjianpeng/Github/kaggle-quora-question-pairs/data/out/2017-04-23_11-51-35/pred/test_311.train_with_swap.pred'

    # 加载预测结果
    test_preds = load_preds(test_preds_fp)
    test_preds = [Model.inverse_adj(y) for y in test_preds]

    # 加载标签文件
    labels = DataUtil.load_vector(cf.get('MODEL', 'train_labels_fp'), True)

    # 加载测试集索引文件
    test_indexs = Feature.load_index(cf.get('MODEL', 'test_indexs_fp'))

    # 获取测试集标签
    test_labels = [labels[index] for index in test_indexs]

    # 评分
    entropy_loss(test_labels, test_preds)

    thresh = 3
    # 设置参数
    feature_name = 'graph_edge_max_clique_size'
    # 特征存储路径
    feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
    train_feature_fp = '%s/%s.train.smat' % (feature_pt, feature_name)
    train_features = Feature.load(train_feature_fp).toarray()
    # 测试集特征
    test_fs = [train_features[index] for index in test_indexs]

    test_labels_l = [test_labels[index] for index in range(len(test_labels)) if test_fs[index] < thresh]
    test_preds_l = [test_preds[index] for index in range(len(test_labels)) if test_fs[index] < thresh]
    entropy_loss(test_labels_l,test_preds_l)

    test_labels_r = [test_labels[index] for index in range(len(test_labels)) if test_fs[index] > thresh]
    test_preds_r = [test_preds[index] for index in range(len(test_labels)) if test_fs[index] > thresh]
    entropy_loss(test_labels_r, test_preds_r)

    test_labels_r = [test_labels[index] for index in range(len(test_labels)) if test_fs[index] == thresh]
    test_preds_r = [test_preds[index] for index in range(len(test_labels)) if test_fs[index] == thresh]
    entropy_loss(test_labels_r, test_preds_r)

if __name__ == "__main__":
    # 读取配置文件
    cf = ConfigParser.ConfigParser()
    cf.read("../conf/python.conf")

    # to_feature_index_run(cf)
    # generate_answer(cf)
    # cal_pos_rate(cf)
    cal_scores()