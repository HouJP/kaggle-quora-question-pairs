#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/6/20 17:43
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


from ..featwheel.extractor import Extractor
import json
import numpy as np


class TreeParser(Extractor):

    @staticmethod
    def init_tree_properties(tree_fp):
        features = {}
        f = open(tree_fp)
        for line in f:
            [qid, json_s] = line.split(' ', 1)
            features[qid] = []
            root = -1
            parent = {}
            indegree = {}
            # calculate in-degree and find father
            if 0 < len(json_s.strip()):
                tree_obj = json.loads(json_s)
                assert len(tree_obj) <= 1
                tree_obj = tree_obj[0]
                for k, r in sorted(tree_obj.items(), key=lambda x: int(x[0]))[1:]:
                    if r['word'] is None:
                        continue
                    head = int(r['head'])
                    k = int(k)
                    if 0 == head:
                        root = k
                    parent[k] = head
                    indegree[head] = indegree.get(head, 0) + 1
            # calculate number of leaves
            n_child = 0
            for i in parent:
                if i not in indegree:
                    n_child += 1
            # calculate the depth of a tree
            depth = 0
            for i in parent:
                if i not in indegree:
                    temp_id = i
                    temp_depth = 0
                    while (temp_id in parent) and (0 != parent[temp_id]):
                        temp_depth += 1
                        temp_id = parent[temp_id]
                    depth = max(depth, temp_depth)
            # calculate the in-degree of root
            n_root_braches = indegree.get(root, 0)
            # calculate the max in-degree
            n_max_braches = 0
            if 0 < len(indegree):
                n_max_braches = max(indegree.values())
            features[str(qid)] = [n_child, depth, n_root_braches, n_max_braches]
        f.close()
        return features

    def __init__(self, config_fp, tree_fp):
        Extractor.__init__(self, config_fp)

        self.tree_properties = TreeParser.init_tree_properties(tree_fp)

    def extract_row(self, row):
        q1_id = str(row['qid1'])
        q2_id = str(row['qid2'])
        q1_features = self.tree_properties[q1_id]
        q2_features = self.tree_properties[q2_id]
        return q1_features + q2_features + abs(np.array(q1_features) - np.array(q2_features)).tolist()

    def get_feature_num(self):
        return 4 * 3


class TreeIndegreeMultiply(Extractor):

    @staticmethod
    def init_tree_ind_multiply(tree_fp):
        features = {}
        f = open(tree_fp)
        for line in f:
            [qid, json_s] = line.split(' ', 1)
            features[qid] = []
            parent = {}
            indegree = {}
            # calculate in-degree and find father node
            if 0 < len(json_s.strip()):
                tree_obj = json.loads(json_s)
                assert len(tree_obj) <= 1
                tree_obj = tree_obj[0]
                for k, r in sorted(tree_obj.items(), key=lambda x: int(x[0]))[1:]:
                    if r['word'] is None:
                        continue
                    head = int(r['head'])
                    k = int(k)
                    parent[k] = head
                    indegree[head] = indegree.get(head, 0) + 1
            # multiply in-degree
            ind_multi = 1.0
            for id_node in indegree:
                ind_multi *= indegree[id_node]
            features[str(qid)] = [ind_multi]
        f.close()
        return features

    def __init__(self, config_fp, tree_fp):
        Extractor.__init__(self, config_fp)

        self.tree_ind_multiply = TreeIndegreeMultiply.init_tree_ind_multiply(tree_fp)

    def extract_row(self, row):
        q1_id = str(row['qid1'])
        q2_id = str(row['qid2'])
        q1_features = self.tree_ind_multiply[q1_id]
        q2_features = self.tree_ind_multiply[q2_id]
        sum_features = (np.array(q1_features) + np.array(q2_features)).tolist()
        sub_features = abs(np.array(q1_features) - np.array(q2_features)).tolist()
        div_features = (np.array(q1_features) / (np.array(q2_features) + 1.)).tolist()
        mul_features = (np.array(q1_features) * (np.array(q2_features) + 0.)).tolist()
        features = q1_features + q2_features + sum_features + sub_features + div_features + mul_features
        return features

    def get_feature_num(self):
        return 6
