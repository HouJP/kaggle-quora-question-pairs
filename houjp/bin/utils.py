# -*- coding: utf-8 -*-
import sys
import re
import time
import random

reload(sys)
sys.setdefaultencoding('utf-8')

class StrUtil(object):
	'''
	字符串工具
	'''

	def __init__(self):
		pass

	@staticmethod
	def tokenize_doc_en(doc):
	    doc = doc.decode('utf-8')	
	    token_pattern = re.compile(r'\b\w\w+\b')
	    lower_doc = doc.lower()
	    tokenize_doc = token_pattern.findall(lower_doc)
	    tokenize_doc = tuple(w for w in tokenize_doc)
	    return tokenize_doc

class LogUtil(object):
	'''
	日志工具
	'''
	def __init__(self):
		pass

	@staticmethod
	def log(typ, msg):
		'''
		打印输出日志信息
		'''
		print "[%s]\t[%s]\t%s" % (TimeUtil.t_now(), typ, str(msg))
		return

class TimeUtil(object):
	'''
	时间工具
	'''
	def __init__(self):
		return

	@staticmethod
	def t_now():
		'''
		返回当前时间，例如"2016-12-27 17:14:01"
		'''
		return time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))

class DataUtil(object):
	'''
	数据集工具
	'''
	def __init__(self):
		return

	@staticmethod
	def save_dic2csv(dic, header, out_fp):
		'''
		存储字典数据结构为CSV格式
		'''
		fout = open(out_fp, 'w')
		fout.write('%s\n' % header)
		for k in dic:
			fout.write('"%s","%s"\n' % (k, dic[k].replace("\"", "\"\"")))
		fout.close()

	# 按比例随机切分数据集
	@staticmethod
	def random_split(instances, rates):
		LogUtil.log("INFO", "random split data(N=%d) into %d parts, with rates(%s) ..." % (len(instances), len(rates), str(rates)))
		slices = []
		pre_sum_rates = []
		sum_rates = 0.0
		for rate in rates:
			slices.append([])
			pre_sum_rates.append(sum_rates + rate)
			sum_rates += rate
		for instance in instances:
			randn = random.random()
			for i in range(0,len(pre_sum_rates)):
				if (randn < pre_sum_rates[i]):
					slices[i].append(instance)
					break
		n_slices = []
		for slic in slices:
			n_slices.append(len(slic))
		LogUtil.log("INFO", "random split data done, with number of instances(%s)." % (str(n_slices)))
		return slices

	# 加载向量
	@staticmethod
	def load_vector(file_path, is_float):
		vector = []
		file = open(file_path)
		for line in file:
			value = float(line.strip()) if is_float else line.strip()
			vector.append(value)
		file.close()
		LogUtil.log("INFO", "load vector done. length=%d" % (len(vector)))
		return vector

	# 存储向量
	@staticmethod
	def save_vector(file_path, vector, mode):
		file = open(file_path, mode)
		for value in vector:
			file.write(str(value) + "\n")
		file.close()
		return

	# 加载矩阵
	@staticmethod
	def load_matrix(file_path):
		matrix = []
		file = open(file_path)
		for line in file:
			vector = line.strip().split(',')
			vector = [ float(vector[i]) for i in range(len(vector)) ]
			matrix.append(vector)
		file.close()
		LogUtil.log("INFO", "load matrix done. size=(%d,%d)" % (len(matrix), len(matrix[0])))
		return matrix

	# 存储矩阵
	@staticmethod
	def save_matrix(file_path, instances, mode):
		file = open(file_path, mode)
		for instance in instances:
			file.write(','.join([ str(instance[i]) for i in range(len(instance)) ]))
			file.write('\n')
		file.close()
		return

# 数学工具
class MathUtil(object):

	@staticmethod
	def count_one_bits(x):
		n = 0
		while (x):
			n += 1 if (x & 0x01) else 0
			x >>= 1
		return n

	@staticmethod
	def int2binarystr(x):
		s = ""
		while (x):
			s += "1" if (x & 0x01) else "0"
			x >>= 1
		return s[::-1]

if __name__ == "__main__":
	# 测试函数
	# print MathUtil.int2binarystr(255)
	# print StrUtil.num2scientificNotation(0.00000000016, -9)
	# print StrUtil.tokenize_doc("he's a teacher 2+34")
	pass