#!/bin/env python
#-*- coding:utf-8 -*-

##############################################################################
#Author:Yxy
#Date:2017-04-26
#Description:Preprocessing smat format feature file
#            Including Normalization and Converting Continuous Features to Bin
#            argv[1] Smat File Name(Path:"./../data/features/question_pair")
#            argv[2] Bin Columns e.g. 0,1,2
##############################################################################

from os.path import isfile
import sys
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import OneHotEncoder

class Preprocess:
    def __init__(self):
        pass

    @staticmethod
    def load(ft_fp):
        """
        WARNING: 很容易造成smat格式与npz格式文件内容不一致
        :param ft_fp:
        :return:
        """
        has_npz = isfile('%s.npz' % ft_fp)
        features = None
        if has_npz:
            features = Preprocess.load_npz(ft_fp)
        else:
            features = Preprocess.load_smat(ft_fp)
        return features

    @staticmethod
    def load_npz(ft_fp):
        loader = np.load('%s.npz' % ft_fp)
        features = csr_matrix((loader['data'],
                           loader['indices'],
                           loader['indptr']),
                          shape=loader['shape'])
        print >> sys.stderr, "INFO", "load npz feature file done (%s)" % ft_fp
        return features

    @staticmethod
    def load_smat(ft_fp):
        '''
        加载特征文件，特征文件格式如下：
        row_num col_num
        f1_index:f1_value f2_index:f2_value ...
        '''
        data = []
        indice = []
        indptr = [0]
        f = open(ft_fp)
        [row_num, col_num] = [int(num) for num in f.readline().strip().split()]
        #t = 0
        for line in f:
            #t += 1
            #if t > 10:
            #    break
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
        features = csr_matrix((data, indice, indptr), shape=(row_num, col_num), dtype=float)
        #features = csr_matrix((data, indice, indptr), shape=(10, col_num), dtype=float)
        print >> sys.stderr, "INFO", "load smat feature file done (%s)" % ft_fp
        return features

    @staticmethod
    def normalization(features, max_param=[], min_param=[], use_param=False):
        if not use_param:
            max_param = features.max(axis=0)
            min_param = features.min(axis=0)
        features_normalization = (features.astype('float64') - min_param) / (max_param - min_param)
        return features_normalization, max_param, min_param

    @staticmethod
    def bin_single_feature(feature, bin_num, index):
        try:
            qbin = pd.qcut(feature, bin_num, range(bin_num), retbins=True)
        except:
            print >> sys.stderr, "Column %d Bin Num %d Failed" %(int(index), int(bin_num))
            qbin = Preprocess.bin_single_feature(feature, bin_num-1, index)
        return qbin

    @staticmethod
    def binfeatures(features, bin_columns, bin_num=10, qbin_param=[], use_param=False):
        features_not_bin = features[:,[x for x in range(features.shape[1]) if x not in bin_columns]]
        matrix_temp = np.zeros((features.shape[0], len(bin_columns)))
        #print features[:,bin_columns]
        if not use_param:
            qbin_param = []
            for x, index in enumerate(bin_columns):
                if type(bin_num) == type([]):
                    qbin = pd.qcut(features[:,index], bin_num, range(len(bin_num)-1), retbins=True)
                else:
                    #qbin = pd.qcut(features[:,index], bin_num, range(bin_num), retbins=True)
                    qbin = Preprocess.bin_single_feature(features[:,index], bin_num, index)
                qbin_param.append(qbin[1])
                matrix_temp[:,x] = qbin[0].get_values()
            #print matrix_temp
        else:
            for x, index in enumerate(bin_columns):
                if type(bin_num) == type([]):
                    bin_num = len(bin_num) - 1
                qbin = np.digitize(features[:,index], qbin_param[x], right=True) - 1
                qbin[qbin>=bin_num] = bin_num - 1
                qbin[qbin<0] = 0 
                matrix_temp[:,x] = qbin
            #print matrix_temp

        #print matrix_temp.shape
        return features_not_bin, matrix_temp, qbin_param
            
    @staticmethod
    def onehotencoder(features, encoder=None, use_param=False):
        #WARNING: if nan not in train but in test maybe cause some error!
        print "Nan Num: ", np.count_nonzero(features != features)
        features[np.isnan(features)] = 0
        if not use_param:
            encoder = OneHotEncoder(handle_unknown="ignore")
            encoder.fit(features)

        features_onehot = encoder.transform(features)
        #print encoder.n_values_
        #print encoder.feature_indices_
        #print features_onehot

        return features_onehot, encoder

    @staticmethod
    def merge_col(matrix1, matrix2):
        return hstack((matrix1, matrix2)).tocsr()

    @staticmethod
    def save_smat(features, ft_pt):
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
        begin_line = True
        for ind_data in range(len(data)):
            while ind_data == indptr[ind_indptr]:
                f.write('\n')
                begin_line = True
                ind_indptr += 1
            if (data[ind_data] < 1e-12) and (data[ind_data] > -1e-12):
                continue
            if (not begin_line) and (ind_data != indptr[ind_indptr - 1]):
                f.write(' ')
            f.write("%d:%.4f" % (indice[ind_data], data[ind_data]))
            begin_line = False
        while ind_indptr < len(indptr):
            f.write("\n")
            ind_indptr += 1
        f.close()

    @staticmethod
    def run(file_name_raw, bin_columns, bin_num=5):
        file_name = "./../../data/feature/question_pair/" + file_name_raw
        train_features_sparse = Preprocess.load("%s.train.smat"%file_name)
        test_features_sparse = Preprocess.load("%s.test.smat"%file_name)

        #CSR format Convert to ndarray format
        train_features = train_features_sparse.toarray()
        test_features = test_features_sparse.toarray()

        #Bin Features
        bin_columns = [] if bin_columns == "" else [int(x) for x in bin_columns.split(",")]
        train_features_nbin, train_features_bin, qbin_param = Preprocess.binfeatures(train_features, bin_columns, bin_num=bin_num)
        test_features_nbin, test_features_bin, qbin_param = Preprocess.binfeatures(test_features, bin_columns, bin_num=bin_num, qbin_param=qbin_param, use_param=True)

        #Feature Normalization
        train_features_nbin_normalization, max_param, min_param = Preprocess.normalization(train_features_nbin)
        test_features_nbin_normalization, max_param, min_param = Preprocess.normalization(test_features_nbin, max_param=max_param, min_param=min_param, use_param=True)

        #Feature OneHotEncoder
        train_features_bin_onehot, onehot_encoder = Preprocess.onehotencoder(train_features_bin)
        test_features_bin_onehot, onehot_encoder = Preprocess.onehotencoder(test_features_bin, encoder=onehot_encoder, use_param=True)

        #Merge Not Bin Cols and Bin Cols
        train_new_features = Preprocess.merge_col(csr_matrix(train_features_nbin_normalization), train_features_bin_onehot)
        test_new_features = Preprocess.merge_col(csr_matrix(test_features_nbin_normalization), test_features_bin_onehot)

        #Save Features
        file_name = "./../../data/feature/question_pair_preprocess/" + file_name_raw
        if type(bin_num) == type([]):
            bin_num = len(bin_num) - 1
        Preprocess.save_smat(train_new_features, "%s_bin%d.train.smat"%(file_name, bin_num))
        Preprocess.save_smat(test_new_features, "%s_bin%d.test.smat"%(file_name, bin_num))

        print "%s Well Done!"%(file_name_raw)

    @staticmethod
    def transform_features():
        #Preprocess.run("ner_match", "2", bin_num=[0.0, 0.6, 0.8, 1])
        #Preprocess.run("LCS_Fea", "2")
        #Preprocess.run("my_word_match_share", "0")
        #Preprocess.run("my_tfidf_word_match_share", "0")
        #Preprocess.run("len_diff", "0")
        #Preprocess.run("len_diff_rate", "0")
        Preprocess.run("AbhisshekFea", "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27")

if __name__ ==  "__main__":
    print len(sys.argv)
    if len(sys.argv) > 1:
        print sys.argv[1], sys.argv[2]
        Preprocess.run(sys.argv[1], sys.argv[2])
    else:
        Preprocess.transform_features()
