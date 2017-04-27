# -*- coding: utf-8 -*-
# ! /usr/bin/python

'''
该脚本中记录的是临时使用的函数
'''

import ConfigParser
import sys

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


if __name__ == "__main__":
    # 读取配置文件
    cf = ConfigParser.ConfigParser()
    cf.read("../conf/python.conf")

    to_feature_index_run(cf)
