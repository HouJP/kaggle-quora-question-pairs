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
from postprocessor import PostProcessor

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


def generate_answer_one(cf, idd, lr, mr, rr):
    # 设置参数
    feature_name = 'graph_edge_max_clique_size'
    # 特征存储路径
    feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
    test_feature_fp = '%s/%s.test.smat' % (feature_pt, feature_name)
    test_features = Feature.load_smat(test_feature_fp).toarray()

    # 设置参数
    feature_name = 'graph_edge_cc_size'
    # 特征存储路径
    feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
    test_feature_fp = '%s/%s.test.smat' % (feature_pt, feature_name)
    test_features_cc = Feature.load_smat(test_feature_fp).toarray()

    thresh = 3
    thresh_cc = 3
    fout = open('/Users/houjianpeng/tmp/%d.csv' % idd, 'w')
    fout.write("\"test_id\",\"is_duplicate\"\n")

    for index in range(len(test_features)):
        if test_features[index][0] >= thresh:
            fout.write('%d,%f\n' % (index, rr))
        elif test_features[index][0] < thresh and test_features_cc[index][0] >= thresh_cc:
            fout.write('%d,%f\n' % (index, mr))
        else:
            fout.write('%d,%f\n' % (index, lr))
    fout.close()


def generate_answer(cf):

    generate_answer_one(cf, 1, 0.29, 0.39, 0.64)
    generate_answer_one(cf, 2, 0.34, 0.55, 0.74)
    generate_answer_one(cf, 3, 0.20, 0.29, 0.84)
    generate_answer_one(cf, 4, 0.13, 0.24, 0.69)
    generate_answer_one(cf, 5, 0.45, 0.66, 0.79)


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

    # 设置参数
    feature_name = 'graph_edge_cc_size'
    # 特征存储路径
    feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
    train_feature_fp = '%s/%s.train.smat' % (feature_pt, feature_name)
    train_features_cc = Feature.load(train_feature_fp).toarray()

    print '-------------------------------------------------'
    print '分析 clique_size <3 / =3 / >3 的各部分：'

    thresh = 3

    len_l = 0
    len_m = 0
    len_r = 0
    len_l_pos = 0
    len_m_pos = 0
    len_r_pos = 0
    for index in range(len(labels)):
        if train_features[index][0] > thresh:
            len_r += 1.
            if labels[index] == 1:
                len_r_pos += 1.
        elif train_features[index][0] == thresh:
            len_m += 1.
            if labels[index] == 1:
                len_m_pos += 1.
        else:
            len_l += 1.
            if labels[index] == 1:
                len_l_pos += 1.
    print 'len_l=%d, len_m=%d, len_r=%d, len_l_pos=%d, len_m_pos=%d, len_r_pos=%d' % (len_l, len_m, len_r, len_l_pos, len_m_pos, len_r_pos)
    print 'rate_l=%f, rate_m=%f, rate_r=%f' % (len_l / len(labels), len_m / len(labels), len_r / len(labels))
    print 'pos_rate_l=%f, pos_rate_m=%f, pos_rate_r=%f' % (len_l_pos / len_l, len_m_pos / len_m, len_r_pos / len_r)

    print '-------------------------------------------------'
    print '分析 clique_size == 2 部分：根据 cc_size 切分为两部分'

    thresh_mc = 3
    thresh_cc = 3

    len_1 = 0
    len_2 = 0
    len_3 = 0
    len_all = 0
    len_pos_1 = 0
    len_pos_2 = 0
    len_pos_3 = 0
    for index in range(len(labels)):
        len_all += 1.
        if train_features[index][0] < thresh_mc:
            if train_features_cc[index][0] < thresh_cc:
                len_1 += 1.
                if labels[index] == 1:
                    len_pos_1 += 1.
            else:
                len_2 += 1.
                if labels[index] == 1:
                    len_pos_2 += 1.
        else:
            len_3 += 1.
            if labels[index] == 1:
                len_pos_3 += 1.
    print 'len_all=%f, len_1=%f(%f), len_2=%f(%f), len_3=%f(%f)' \
          % (len_all, len_1, 1.0 * len_1 / len_all, len_2, 1.0 * len_2 / len_all, len_3, 1.0 * len_3 / len_all)
    print 'pos_1=%f, pos_2=%f, pos_3=%f' % (1.0 * len_pos_1 / len_1, 1.0 * len_pos_2 / len_2, 1. * len_pos_3 / len_3)


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


def cal_scores(argv):
    test_preds_fp = argv[0]
    # test_preds_fp = '/Users/houjianpeng/Github/kaggle-quora-question-pairs/data/out/2017-05-03_11-27-48/pred/test_311.train_with_swap.pred' # v2_20_9
    # test_preds_fp = '/Users/houjianpeng/tmp/test_311.train_with_swap.pred'

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

    thresh_cc = 3
    # 设置参数
    feature_name = 'graph_edge_cc_size'
    # 特征存储路径
    feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
    train_feature_fp = '%s/%s.train.smat' % (feature_pt, feature_name)
    train_features_cc = Feature.load(train_feature_fp).toarray()
    test_fs_cc = [train_features_cc[index] for index in test_indexs]

    print '-------------------------------------------------'
    print '分析 clique_size <3 / =3 / >3 的各部分得分：'
    test_labels_l = [test_labels[index] for index in range(len(test_labels)) if test_fs[index] < thresh]
    test_preds_l = [test_preds[index] for index in range(len(test_labels)) if test_fs[index] < thresh]
    entropy_loss(test_labels_l,test_preds_l)
    LogUtil.log('INFO', 'rate_labels_l=%f, rate_preds_l=%f' % (1. * sum(test_labels_l) / len(test_labels_l), 1. * sum(test_preds_l) / len(test_preds_l)))

    test_labels_m = [test_labels[index] for index in range(len(test_labels)) if test_fs[index] == thresh]
    test_preds_m = [test_preds[index] for index in range(len(test_labels)) if test_fs[index] == thresh]
    entropy_loss(test_labels_m, test_preds_m)
    LogUtil.log('INFO', 'rate_labels_m=%f, rate_preds_m=%f' % (
        1. * sum(test_labels_m) / len(test_labels_m), 1. * sum(test_preds_m) / len(test_preds_m)))

    test_labels_r = [test_labels[index] for index in range(len(test_labels)) if test_fs[index] > thresh]
    test_preds_r = [test_preds[index] for index in range(len(test_labels)) if test_fs[index] > thresh]
    entropy_loss(test_labels_r, test_preds_r)
    LogUtil.log('INFO', 'rate_labels_r=%f, rate_preds_r=%f' % (
        1. * sum(test_labels_r) / len(test_labels_r), 1. * sum(test_preds_r) / len(test_preds_r)))

    print '-------------------------------------------------'
    print '分析 clique_size <3 部分得分，根据 cc_size 切分为两部分：'

    test_labels_1 = [test_labels[index] for index in range(len(test_labels)) if (test_fs[index] < thresh and test_fs_cc[index] < thresh)]
    test_preds_1 = [test_preds[index] for index in range(len(test_labels)) if (test_fs[index] < thresh and test_fs_cc[index] < thresh)]
    entropy_loss(test_labels_1, test_preds_1)
    LogUtil.log('INFO', 'rate_labels_1=%f, rate_preds_1=%f' % (
    1. * sum(test_labels_1) / len(test_labels_1), 1. * sum(test_preds_1) / len(test_preds_1)))

    test_labels_2 = [test_labels[index] for index in range(len(test_labels)) if
                     (test_fs[index] < thresh and test_fs_cc[index] >= thresh)]
    test_preds_2 = [test_preds[index] for index in range(len(test_labels)) if
                    (test_fs[index] < thresh and test_fs_cc[index] >= thresh)]
    entropy_loss(test_labels_2, test_preds_2)
    LogUtil.log('INFO', 'rate_labels_2=%f, rate_preds_2=%f' % (
        1. * sum(test_labels_2) / len(test_labels_2), 1. * sum(test_preds_2) / len(test_preds_2)))

def part_answer(cf):
    # 加载预测结果
    test_preds_fp = '/Users/houjianpeng/tmp/v4_35_10.pred'
    test_preds = PostProcessor.read_result_list(test_preds_fp)
    # test_preds = [Model.inverse_adj(y) for y in test_preds]
    LogUtil.log('INFO', 'len(test_preds)=%d' % len(test_preds))

    thresh = 3

    # 加载特征
    feature_name = 'graph_edge_max_clique_size'
    feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
    test_feature_fp = '%s/%s.test.smat' % (feature_pt, feature_name)
    test_features = Feature.load(test_feature_fp).toarray()
    LogUtil.log('INFO', 'len(test_features)=%d' % len(test_features))

    count = 0.
    for index in range(len(test_preds)):
        score = test_preds[index]
        if test_features[index] == 3.:
            count += 1.
            score = score  # Model.adj(score, te=0.40883512)#, tr=0.623191)
        elif test_features[index] > 3.:
            score = 0.5  # Model.adj(score, te=0.96503024)#, tr=0.972554)
        else:
            score = 0.5  # Model.adj(score, te=0.04957855)#, tr=0.183526)
        test_preds[index] = score
    LogUtil.log('INFO', 'count=%d' % count)

    fout = open('/Users/houjianpeng/tmp/part_answer_v4_35_10.csv', 'w')
    fout.write("\"test_id\",\"is_duplicate\"\n")

    for index in range(len(test_preds)):
        fout.write('%d,%s\n' % (index, test_preds[index]))
    fout.close()

    # 设置参数
    feature_name = 'graph_edge_max_clique_size'
    # 特征存储路径
    feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
    test_feature_fp = '%s/%s.test.smat' % (feature_pt, feature_name)
    test_features_mc = Feature.load(test_feature_fp).toarray()

    # 设置参数
    feature_name = 'graph_edge_cc_size'
    # 特征存储路径
    feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
    test_feature_fp = '%s/%s.test.smat' % (feature_pt, feature_name)
    test_features_cc = Feature.load(test_feature_fp).toarray()

    print '-------------------------------------------------'
    print '分析 clique_size <3 / =3 / >3 的各部分：'

    thresh = 3

    len_l = 0
    len_m = 0
    len_r = 0
    len_l_pos = 0
    len_m_pos = 0
    len_r_pos = 0
    for index in range(len(test_preds)):
        if test_features[index][0] > thresh:
            len_r += 1.
            len_r_pos += test_preds[index]
        elif test_features[index][0] == thresh:
            len_m += 1.
            len_m_pos += test_preds[index]
        else:
            len_l += 1.
            len_l_pos += test_preds[index]
    print 'len_l=%d, len_m=%d, len_r=%d, len_l_pos=%d, len_m_pos=%d, len_r_pos=%d' % (
    len_l, len_m, len_r, len_l_pos, len_m_pos, len_r_pos)
    print 'rate_l=%f, rate_m=%f, rate_r=%f' % (len_l / len(test_preds), len_m / len(test_preds), len_r / len(test_preds))
    print 'pos_rate_l=%f, pos_rate_m=%f, pos_rate_r=%f' % (len_l_pos / len_l, len_m_pos / len_m, len_r_pos / len_r)

    print '-------------------------------------------------'
    print '分析 clique_size == 2 部分：根据 cc_size 切分为两部分'

    thresh_mc = 3
    thresh_cc = 3

    len_1 = 0
    len_2 = 0
    len_3 = 0
    len_all = 0
    len_pos_1 = 0
    len_pos_2 = 0
    len_pos_3 = 0
    for index in range(len(test_preds)):
        len_all += 1.
        if test_features[index][0] < thresh_mc:
            if test_features_cc[index][0] < thresh_cc:
                len_1 += 1.
                len_pos_1 += test_preds[index]
            else:
                len_2 += 1.
                len_pos_2 += test_preds[index]
        else:
            len_3 += 1.
            len_pos_3 += test_preds[index]
    print 'len_all=%f, len_1=%f(%f), len_2=%f(%f), len_3=%f(%f)' \
          % (len_all, len_1, 1.0 * len_1 / len_all, len_2, 1.0 * len_2 / len_all, len_3, 1.0 * len_3 / len_all)
    print 'pos_1=%f, pos_2=%f, pos_3=%f' % (1.0 * len_pos_1 / len_1, 1.0 * len_pos_2 / len_2, 1. * len_pos_3 / len_3)



def rescale_answer(cf):
    # 加载预测结果
    test_preds_fp = '/Users/houjianpeng/tmp/merge_2/xgb_v4_55_10_lgb_unkown.online.pred'
    test_preds = PostProcessor.read_result_list(test_preds_fp)
    test_preds = [Model.inverse_adj(y) for y in test_preds]
    LogUtil.log('INFO', 'len(test_preds)=%d' % len(test_preds))

    thresh = 3

    # 加载特征
    feature_name = 'graph_edge_max_clique_size'
    feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
    test_feature_fp = '%s/%s.test.smat' % (feature_pt, feature_name)
    test_features = Feature.load(test_feature_fp).toarray()
    LogUtil.log('INFO', 'len(test_features)=%d' % len(test_features))

    count = 0.
    for index in range(len(test_preds)):
        score = test_preds[index]
        if test_features[index] == 3.:
            count += 1.
            score = Model.adj(score, te=0.40883512, tr=0.623191)
        elif test_features[index] > 3.:
            score = Model.adj(score, te=0.96503024, tr=0.972554)
        else:
            score = Model.adj(score, te=0.04957855, tr=0.183526)
        test_preds[index] = score
    LogUtil.log('INFO', 'count=%d' % count)

    fout = open('/Users/houjianpeng/tmp/merge_2/rescale_xgb_v4_55_10_lgb_unkown.online.pred', 'w')
    fout.write("\"test_id\",\"is_duplicate\"\n")

    for index in range(len(test_preds)):
        fout.write('%d,%s\n' % (index, test_preds[index]))
    fout.close()

    # 设置参数
    feature_name = 'graph_edge_max_clique_size'
    # 特征存储路径
    feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
    test_feature_fp = '%s/%s.test.smat' % (feature_pt, feature_name)
    test_features_mc = Feature.load(test_feature_fp).toarray()

    # 设置参数
    feature_name = 'graph_edge_cc_size'
    # 特征存储路径
    feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
    test_feature_fp = '%s/%s.test.smat' % (feature_pt, feature_name)
    test_features_cc = Feature.load(test_feature_fp).toarray()

    print '-------------------------------------------------'
    print '分析 clique_size <3 / =3 / >3 的各部分：'

    thresh = 3

    len_l = 0
    len_m = 0
    len_r = 0
    len_l_pos = 0
    len_m_pos = 0
    len_r_pos = 0
    for index in range(len(test_preds)):
        if test_features[index][0] > thresh:
            len_r += 1.
            len_r_pos += test_preds[index]
        elif test_features[index][0] == thresh:
            len_m += 1.
            len_m_pos += test_preds[index]
        else:
            len_l += 1.
            len_l_pos += test_preds[index]
    print 'len_l=%d, len_m=%d, len_r=%d, len_l_pos=%d, len_m_pos=%d, len_r_pos=%d' % (
    len_l, len_m, len_r, len_l_pos, len_m_pos, len_r_pos)
    print 'rate_l=%f, rate_m=%f, rate_r=%f' % (len_l / len(test_preds), len_m / len(test_preds), len_r / len(test_preds))
    print 'pos_rate_l=%f, pos_rate_m=%f, pos_rate_r=%f' % (len_l_pos / len_l, len_m_pos / len_m, len_r_pos / len_r)

    print '-------------------------------------------------'
    print '分析 clique_size == 2 部分：根据 cc_size 切分为两部分'

    thresh_mc = 3
    thresh_cc = 3

    len_1 = 0
    len_2 = 0
    len_3 = 0
    len_all = 0
    len_pos_1 = 0
    len_pos_2 = 0
    len_pos_3 = 0
    for index in range(len(test_preds)):
        len_all += 1.
        if test_features[index][0] < thresh_mc:
            if test_features_cc[index][0] < thresh_cc:
                len_1 += 1.
                len_pos_1 += test_preds[index]
            else:
                len_2 += 1.
                len_pos_2 += test_preds[index]
        else:
            len_3 += 1.
            len_pos_3 += test_preds[index]
    print 'len_all=%f, len_1=%f(%f), len_2=%f(%f), len_3=%f(%f)' \
          % (len_all, len_1, 1.0 * len_1 / len_all, len_2, 1.0 * len_2 / len_all, len_3, 1.0 * len_3 / len_all)
    print 'pos_1=%f, pos_2=%f, pos_3=%f' % (1.0 * len_pos_1 / len_1, 1.0 * len_pos_2 / len_2, 1. * len_pos_3 / len_3)

def rescale2_answer(cf):
    # 加载预测结果
    te = 0.173
    tr = 0.369
    has_postprocess = True
    test_preds_fp = '/Users/houjianpeng/tmp/v4_215_17_tag65/cv_n5_online.test.pred'
    fout = open('/Users/houjianpeng/tmp/v4_215_17_tag65/rescale_cv_n5_online.test.pred', 'w')
    test_preds = PostProcessor.read_result_list(test_preds_fp)
    if has_postprocess:
        test_preds = [Model.inverse_adj(y,te=te,tr=tr) for y in test_preds]
    LogUtil.log('INFO', 'len(test_preds)=%d' % len(test_preds))

    thresh = 3

    # 设置参数
    feature_name = 'graph_edge_max_clique_size'
    # 特征存储路径
    feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
    test_feature_fp = '%s/%s.test.smat' % (feature_pt, feature_name)
    test_features_mc = Feature.load(test_feature_fp).toarray()

    # 设置参数
    feature_name = 'graph_edge_cc_size'
    # 特征存储路径
    feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
    test_feature_fp = '%s/%s.test.smat' % (feature_pt, feature_name)
    test_features_cc = Feature.load(test_feature_fp).toarray()

    print '-------------------------------------------------'
    print '缩放答案：'

    for index in range(len(test_preds)):
        score = test_preds[index]
        if test_features_mc[index][0] == 3.:
            # score = Model.adj(score, te=0.40883512, tr=0.459875)
            score = Model.adj(score, te=0.40883512, tr=0.623191)
        elif test_features_mc[index][0] > 3.:
            # score = Model.adj(score, te=0.96503024, tr=0.971288)
            score = Model.adj(score, te=0.96503024, tr=0.972554)
        else:
            if test_features_cc[index][0] < 3.:
                # score = Model.adj(score, te=0.05739666, tr=0.101436)
                score = Model.adj(score, te=0.05739666, tr=0.233473)
            else:
                # score = Model.adj(score, te=0.04503431, tr=0.093469)
                score = Model.adj(score, te=0.04503431, tr=0.149471)
        test_preds[index] = score


    fout.write("\"test_id\",\"is_duplicate\"\n")

    for index in range(len(test_preds)):
        fout.write('%d,%s\n' % (index, test_preds[index]))
    fout.close()

    print '-------------------------------------------------'
    print '分析 clique_size <3 / =3 / >3 的各部分：'

    thresh = 3

    len_l = 0
    len_m = 0
    len_r = 0
    len_l_pos = 0
    len_m_pos = 0
    len_r_pos = 0
    for index in range(len(test_preds)):
        if test_features_mc[index][0] > thresh:
            len_r += 1.
            len_r_pos += test_preds[index]
        elif test_features_mc[index][0] == thresh:
            len_m += 1.
            len_m_pos += test_preds[index]
        else:
            len_l += 1.
            len_l_pos += test_preds[index]
    print 'len_l=%d, len_m=%d, len_r=%d, len_l_pos=%d, len_m_pos=%d, len_r_pos=%d' % (
    len_l, len_m, len_r, len_l_pos, len_m_pos, len_r_pos)
    print 'rate_l=%f, rate_m=%f, rate_r=%f' % (len_l / len(test_preds), len_m / len(test_preds), len_r / len(test_preds))
    print 'pos_rate_l=%f, pos_rate_m=%f, pos_rate_r=%f' % (len_l_pos / len_l, len_m_pos / len_m, len_r_pos / len_r)

    print '-------------------------------------------------'
    print '分析 clique_size == 2 部分：根据 cc_size 切分为两部分'

    thresh_mc = 3
    thresh_cc = 3

    len_1 = 0
    len_2 = 0
    len_3 = 0
    len_all = 0
    len_pos_1 = 0
    len_pos_2 = 0
    len_pos_3 = 0
    for index in range(len(test_preds)):
        len_all += 1.
        if test_features_mc[index][0] < thresh_mc:
            if test_features_cc[index][0] < thresh_cc:
                len_1 += 1.
                len_pos_1 += test_preds[index]
            else:
                len_2 += 1.
                len_pos_2 += test_preds[index]
        else:
            len_3 += 1.
            len_pos_3 += test_preds[index]
    print 'len_all=%f, len_1=%f(%f), len_2=%f(%f), len_3=%f(%f)' \
          % (len_all, len_1, 1.0 * len_1 / len_all, len_2, 1.0 * len_2 / len_all, len_3, 1.0 * len_3 / len_all)
    print 'pos_1=%f, pos_2=%f, pos_3=%f' % (1.0 * len_pos_1 / len_1, 1.0 * len_pos_2 / len_2, 1. * len_pos_3 / len_3)

def rescale3_answer(cf):
    # 加载预测结果
    te = 0.173
    tr = 0.43121
    test_preds_fp = '/Users/houjianpeng/tmp/v5_125_10/full.test.pred'
    fout = open('/Users/houjianpeng/tmp/v5_125_10/rescale_full.test.pred', 'w')
    test_preds = PostProcessor.read_result_list(test_preds_fp)
    test_preds = [Model.inverse_adj(y, te=te, tr=tr) for y in test_preds]
    LogUtil.log('INFO', 'len(test_preds)=%d' % len(test_preds))

    thresh = 3

    # 设置参数
    feature_name = 'graph_edge_max_clique_size'
    # 特征存储路径
    feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
    test_feature_fp = '%s/%s.test.smat' % (feature_pt, feature_name)
    test_features_mc = Feature.load(test_feature_fp).toarray()

    # 设置参数
    feature_name = 'graph_edge_cc_size'
    # 特征存储路径
    feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
    test_feature_fp = '%s/%s.test.smat' % (feature_pt, feature_name)
    test_features_cc = Feature.load(test_feature_fp).toarray()

    print '-------------------------------------------------'
    print '缩放答案：'

    for index in range(len(test_preds)):
        score = test_preds[index]
        if test_features_mc[index][0] == 3.:
            # score = Model.adj(score, te=0.40883512, tr=0.459875)
            score = Model.adj(score, te=0.40883512, tr=0.623191)
        elif test_features_mc[index][0] > 3.:
            # score = Model.adj(score, te=0.96503024, tr=0.971288)
            # score = Model.adj(score, te=0.96503024, tr=0.972554)
            score = Model.adj(score, te=0.96503024, tr=0.9829238325818231)
        else:
            if test_features_cc[index][0] < 3.:
                # score = Model.adj(score, te=0.05739666, tr=0.101436)
                score = Model.adj(score, te=0.05739666, tr=0.233473)
            else:
                # score = Model.adj(score, te=0.04503431, tr=0.093469)
                score = Model.adj(score, te=0.04503431, tr=0.149471)
        test_preds[index] = score


    fout.write("\"test_id\",\"is_duplicate\"\n")

    for index in range(len(test_preds)):
        fout.write('%d,%s\n' % (index, test_preds[index]))
    fout.close()

    print '-------------------------------------------------'
    print '分析 clique_size <3 / =3 / >3 的各部分：'

    thresh = 3

    len_l = 0
    len_m = 0
    len_r = 0
    len_l_pos = 0
    len_m_pos = 0
    len_r_pos = 0
    for index in range(len(test_preds)):
        if test_features_mc[index][0] > thresh:
            len_r += 1.
            len_r_pos += test_preds[index]
        elif test_features_mc[index][0] == thresh:
            len_m += 1.
            len_m_pos += test_preds[index]
        else:
            len_l += 1.
            len_l_pos += test_preds[index]
    print 'len_l=%d, len_m=%d, len_r=%d, len_l_pos=%d, len_m_pos=%d, len_r_pos=%d' % (
    len_l, len_m, len_r, len_l_pos, len_m_pos, len_r_pos)
    print 'rate_l=%f, rate_m=%f, rate_r=%f' % (len_l / len(test_preds), len_m / len(test_preds), len_r / len(test_preds))
    print 'pos_rate_l=%f, pos_rate_m=%f, pos_rate_r=%f' % (len_l_pos / len_l, len_m_pos / len_m, len_r_pos / len_r)

    print '-------------------------------------------------'
    print '分析 clique_size == 2 部分：根据 cc_size 切分为两部分'

    thresh_mc = 3
    thresh_cc = 3

    len_1 = 0
    len_2 = 0
    len_3 = 0
    len_all = 0
    len_pos_1 = 0
    len_pos_2 = 0
    len_pos_3 = 0
    for index in range(len(test_preds)):
        len_all += 1.
        if test_features_mc[index][0] < thresh_mc:
            if test_features_cc[index][0] < thresh_cc:
                len_1 += 1.
                len_pos_1 += test_preds[index]
            else:
                len_2 += 1.
                len_pos_2 += test_preds[index]
        else:
            len_3 += 1.
            len_pos_3 += test_preds[index]
    print 'len_all=%f, len_1=%f(%f), len_2=%f(%f), len_3=%f(%f)' \
          % (len_all, len_1, 1.0 * len_1 / len_all, len_2, 1.0 * len_2 / len_all, len_3, 1.0 * len_3 / len_all)
    print 'pos_1=%f, pos_2=%f, pos_3=%f' % (1.0 * len_pos_1 / len_1, 1.0 * len_pos_2 / len_2, 1. * len_pos_3 / len_3)


class Grocery(object):

    @staticmethod
    def std_rescale_answer(cf, test_preds_fp):
        # 加载预测结果
        te = float(cf.get('MODEL', 'te'))
        tr = float(cf.get('MODEL', 'tr'))
        fout = open(test_preds_fp + '.rescale', 'w')
        test_preds = PostProcessor.read_result_list(test_preds_fp)
        if cf.get('MODEL', 'has_postprocess') == 'True':
            test_preds = [Model.inverse_adj(y, te=te, tr=tr) for y in test_preds]
        LogUtil.log('INFO', 'len(test_preds)=%d' % len(test_preds))

        thresh = 3

        # 设置参数
        feature_name = 'graph_edge_max_clique_size'
        # 特征存储路径
        feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
        test_feature_fp = '%s/%s.test.smat' % (feature_pt, feature_name)
        test_features_mc = Feature.load(test_feature_fp).toarray()

        # 设置参数
        feature_name = 'graph_edge_cc_size'
        # 特征存储路径
        feature_pt = cf.get('DEFAULT', 'feature_question_pair_pt')
        test_feature_fp = '%s/%s.test.smat' % (feature_pt, feature_name)
        test_features_cc = Feature.load(test_feature_fp).toarray()

        print '-------------------------------------------------'
        print '缩放答案：'

        for index in range(len(test_preds)):
            score = test_preds[index]
            if test_features_mc[index][0] == 3.:
                # score = Model.adj(score, te=0.40883512, tr=0.459875)
                score = Model.adj(score, te=0.40883512, tr=0.623191)
            elif test_features_mc[index][0] > 3.:
                # score = Model.adj(score, te=0.96503024, tr=0.971288)
                score = Model.adj(score, te=0.96503024, tr=0.972554)
            else:
                if test_features_cc[index][0] < 3.:
                    # score = Model.adj(score, te=0.05739666, tr=0.101436)
                    score = Model.adj(score, te=0.05739666, tr=0.233473)
                else:
                    # score = Model.adj(score, te=0.04503431, tr=0.093469)
                    score = Model.adj(score, te=0.04503431, tr=0.149471)
            test_preds[index] = score


        fout.write("\"test_id\",\"is_duplicate\"\n")

        for index in range(len(test_preds)):
            fout.write('%d,%s\n' % (index, test_preds[index]))
        fout.close()

        print '-------------------------------------------------'
        print '分析 clique_size <3 / =3 / >3 的各部分：'

        thresh = 3

        len_l = 0
        len_m = 0
        len_r = 0
        len_l_pos = 0
        len_m_pos = 0
        len_r_pos = 0
        for index in range(len(test_preds)):
            if test_features_mc[index][0] > thresh:
                len_r += 1.
                len_r_pos += test_preds[index]
            elif test_features_mc[index][0] == thresh:
                len_m += 1.
                len_m_pos += test_preds[index]
            else:
                len_l += 1.
                len_l_pos += test_preds[index]
        print 'len_l=%d, len_m=%d, len_r=%d, len_l_pos=%d, len_m_pos=%d, len_r_pos=%d' % (
        len_l, len_m, len_r, len_l_pos, len_m_pos, len_r_pos)
        print 'rate_l=%f, rate_m=%f, rate_r=%f' % (len_l / len(test_preds), len_m / len(test_preds), len_r / len(test_preds))
        print 'pos_rate_l=%f, pos_rate_m=%f, pos_rate_r=%f' % (len_l_pos / len_l, len_m_pos / len_m, len_r_pos / len_r)

        print '-------------------------------------------------'
        print '分析 clique_size == 2 部分：根据 cc_size 切分为两部分'

        thresh_mc = 3
        thresh_cc = 3

        len_1 = 0
        len_2 = 0
        len_3 = 0
        len_all = 0
        len_pos_1 = 0
        len_pos_2 = 0
        len_pos_3 = 0
        for index in range(len(test_preds)):
            len_all += 1.
            if test_features_mc[index][0] < thresh_mc:
                if test_features_cc[index][0] < thresh_cc:
                    len_1 += 1.
                    len_pos_1 += test_preds[index]
                else:
                    len_2 += 1.
                    len_pos_2 += test_preds[index]
            else:
                len_3 += 1.
                len_pos_3 += test_preds[index]
        print 'len_all=%f, len_1=%f(%f), len_2=%f(%f), len_3=%f(%f)' \
              % (len_all, len_1, 1.0 * len_1 / len_all, len_2, 1.0 * len_2 / len_all, len_3, 1.0 * len_3 / len_all)
        print 'pos_1=%f, pos_2=%f, pos_3=%f' % (1.0 * len_pos_1 / len_1, 1.0 * len_pos_2 / len_2, 1. * len_pos_3 / len_3)

if __name__ == "__main__":
    # 读取配置文件
    cf = ConfigParser.ConfigParser()
    # cf.read(sys.argv[1])
    cf.read('../conf/python.conf')

    # to_feature_index_run(cf)
    # generate_answer(cf)
    # cal_pos_rate(cf)
    # cal_scores(sys.argv[2:])
    rescale2_answer(cf)
    # part_answer(cf)