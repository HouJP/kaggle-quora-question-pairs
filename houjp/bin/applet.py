# -*- coding: utf-8 -*-
#! /usr/bin/python

import ConfigParser
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

def toCSV(in_fp, out_fp, header):
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

def toCSVRun(cf):
	devel_pt = cf.get('path', 'devel_pt')

	in_fp = "%s/relation.devel-311.train.txt" % devel_pt
	out_fp = "%s/relation.devel-311.train.csv" % devel_pt
	header = "\"is_duplicate\",\"qid1\",\"qid2\""
	toCSV(in_fp, out_fp, header)

	in_fp = "%s/relation.devel-311.valid.txt" % devel_pt
	out_fp = "%s/relation.devel-311.valid.csv" % devel_pt
	header = "\"is_duplicate\",\"qid1\",\"qid2\""
	toCSV(in_fp, out_fp, header)

	in_fp = "%s/relation.devel-311.test.txt" % devel_pt
	out_fp = "%s/relation.devel-311.test.csv" % devel_pt
	header = "\"is_duplicate\",\"qid1\",\"qid2\""
	toCSV(in_fp, out_fp, header)


	in_fp = "%s/relation.devel-811.train.txt" % devel_pt
	out_fp = "%s/relation.devel-811.train.csv" % devel_pt
	header = "\"is_duplicate\",\"qid1\",\"qid2\""
	toCSV(in_fp, out_fp, header)

	in_fp = "%s/relation.devel-811.valid.txt" % devel_pt
	out_fp = "%s/relation.devel-811.valid.csv" % devel_pt
	header = "\"is_duplicate\",\"qid1\",\"qid2\""
	toCSV(in_fp, out_fp, header)

	in_fp = "%s/relation.devel-811.test.txt" % devel_pt
	out_fp = "%s/relation.devel-811.test.csv" % devel_pt
	header = "\"is_duplicate\",\"qid1\",\"qid2\""
	toCSV(in_fp, out_fp, header)

	return

if __name__ == "__main__":
	# 读取配置文件
	cf = ConfigParser.ConfigParser()
	cf.read("../conf/python.conf")

	toCSV_run(cf)

