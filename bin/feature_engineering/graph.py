#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/6/16 20:07
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


import csv

import networkx as nx

from bin.featwheel.utils import LogUtil
from bin.featwheel.utils import MathUtil
from ..featwheel.feature import Feature
from ..featwheel.extractor import Extractor


class Graph(Extractor):
    @staticmethod
    def generate_graph_clique(G):
        n2clique = {}
        cliques = []
        for clique in nx.find_cliques(G):
            for n in clique:
                if n not in n2clique:
                    n2clique[n] = []
                n2clique[n].append(len(cliques))
            cliques.append(clique)
        LogUtil.log('INFO', 'init graph cliques done, len(cliques)=%d' % len(cliques))
        return n2clique, cliques

    @staticmethod
    def generate_graph_cc(G):
        n2cc = {}
        ccs = []
        for cc in nx.connected_components(G):
            for n in cc:
                if n in n2cc:
                    LogUtil.log('WARNING', '%d already in n2cc(=%d)' % (n, n2cc[n]))
                n2cc[n] = len(ccs)
            ccs.append(cc)
        LogUtil.log('INFO', 'init graph cc done, len(cliques)=%d' % len(ccs))
        return n2cc, ccs

    @staticmethod
    def generate_pagerank(G, alpha, max_iter):
        pr = nx.pagerank(G, alpha=alpha, max_iter=max_iter)
        LogUtil.log('INFO', 'Graph cal pagerank done')
        return pr

    @staticmethod
    def generate_hits(G, max_iter):
        hits_h, hits_a = nx.hits(G, max_iter=max_iter)
        LogUtil.log('INFO', 'Graph cal hits done')
        return hits_h, hits_a


class DG(Graph):
    pass


class UDG(Graph):
    @staticmethod
    def generate_graph(config, weight_feature_name, weight_feature_id, reverse):
        q2id = {}
        e2weight = {}
        G = nx.Graph()

        train_wfs_fs = None
        test_wfs_fs = None
        if weight_feature_name is not None:
            train_wfs_fs = Feature.load(
                '%s/%s.train.smat' % (config.get('DIRECTORY', 'feature_question_pair_pt'), weight_feature_name)).toarray()
            test_wfs_fs = Feature.load(
                '%s/%s.test.smat' % (config.get('DIRECTORY', 'feature_question_pair_pt'), weight_feature_name)).toarray()
            if 'True' == reverse:
                LogUtil.log('INFO', 'will reverse')
                for index in range(len(train_wfs_fs)):
                    train_wfs_fs[index][weight_feature_id] = 1. - train_wfs_fs[index][weight_feature_id]
                for index in range(len(test_wfs_fs)):
                    test_wfs_fs[index][weight_feature_id] = 1. - test_wfs_fs[index][weight_feature_id]

        fin = csv.reader(open('%s/train.csv' % config.get('DIRECTORY', 'origin_pt')))
        fin.next()
        index = 0
        for p in fin:
            q1 = str(p[3]).strip()
            q2 = str(p[4]).strip()
            weight = 0 if train_wfs_fs is None else train_wfs_fs[index][weight_feature_id]
            if q1 not in q2id:
                q2id[q1] = len(q2id)
            if q2 not in q2id:
                q2id[q2] = len(q2id)
            G.add_edge(q2id[q1], q2id[q2], weight=weight)
            e2weight[(q2id[q1], q2id[q2])] = weight
            e2weight[(q2id[q2], q2id[q1])] = weight
            index += 1

        fin = csv.reader(open('%s/test.csv' % config.get('DIRECTORY', 'origin_pt')))
        fin.next()
        index = 0
        for p in fin:
            q1 = str(p[1]).strip()
            q2 = str(p[2]).strip()
            weight = 0 if test_wfs_fs is None else test_wfs_fs[index][weight_feature_id]
            if q1 not in q2id:
                q2id[q1] = len(q2id)
            if q2 not in q2id:
                q2id[q2] = len(q2id)
            G.add_edge(q2id[q1], q2id[q2], weight=weight)
            e2weight[(q2id[q1], q2id[q2])] = weight
            e2weight[(q2id[q2], q2id[q1])] = weight
            index += 1
        LogUtil.log('INFO', 'Graph constructed.')

        return q2id, e2weight, G

    def __init__(self, config_fp, weight_feature_name=None, weight_feature_id=None, reverse=False):
        Extractor.__init__(self, config_fp)
        self.feature_name = '%s_%s_%s_%s' % (self.__class__.__name__,
                                             str(weight_feature_name),
                                             str(weight_feature_id),
                                             str(reverse))
        self.q2id, self.e2weight, self.G = UDG.generate_graph(self.config,
                                                              weight_feature_name,
                                                              weight_feature_id,
                                                              reverse)


class GraphEdgeMaxCliqueSize(UDG):
    """
    Feature: max clique size of the edge
    """
    def __init__(self, config_fp):
        UDG.__init__(self, config_fp)
        # extract clique from graph
        self.n2clique, self.cliques = Graph.generate_graph_clique(self.G)

    def get_feature_num(self):
        return 1

    def extract_row(self, row):
        q1 = str(row['question1']).strip()
        q2 = str(row['question2']).strip()

        qid1 = self.q2id[q1]
        qid2 = self.q2id[q2]

        edge_max_clique_size = 0

        for clique_id in self.n2clique[qid1]:
            if qid2 in self.cliques[clique_id]:
                edge_max_clique_size = max(edge_max_clique_size, len(self.cliques[clique_id]))

        return [edge_max_clique_size]


class GraphNodeMaxCliqueSize(GraphEdgeMaxCliqueSize):
    def extract_row(self, row):
        q1 = str(row['question1']).strip()
        q2 = str(row['question2']).strip()
        qid1 = self.q2id[q1]
        qid2 = self.q2id[q2]
        lnode_max_clique_size = 0
        rnode_max_clique_size = 0
        for clique_id in self.n2clique[qid1]:
            lnode_max_clique_size = max(lnode_max_clique_size, len(self.cliques[clique_id]))

        for clique_id in self.n2clique[qid2]:
            rnode_max_clique_size = max(rnode_max_clique_size, len(self.cliques[clique_id]))

        return [lnode_max_clique_size,
                rnode_max_clique_size,
                max(lnode_max_clique_size, rnode_max_clique_size),
                min(lnode_max_clique_size, rnode_max_clique_size)]

    def get_feature_num(self):
        return 4


class GraphNumClique(UDG):
    def __init__(self, config_fp):
        UDG.__init__(self, config_fp)
        # extract clique from graph
        self.n2clique, self.cliques = Graph.generate_graph_clique(self.G)

    def extract_row(self, row):
        q1 = str(row['question1']).strip()
        q2 = str(row['question2']).strip()
        qid1 = self.q2id[q1]
        qid2 = self.q2id[q2]

        num_clique = 0
        for clique_id in self.n2clique[qid1]:
            if qid2 in self.cliques[clique_id]:
                num_clique += 1
        return [num_clique]

    def get_feature_num(self):
        return 1


class GraphEdgeCCSize(UDG):
    def __init__(self, config_fp):
        UDG.__init__(self, config_fp)
        self.n2cc, self.ccs = Graph.generate_graph_cc(self.G)

    def extract_row(self, row):
        q1 = str(row['question1']).strip()
        qid1 = self.q2id[q1]
        edge_cc_size = len(self.ccs[self.n2cc[qid1]])
        return [edge_cc_size]

    def get_feature_num(self):
        return 1


class GraphPagerank(UDG):
    def __init__(self, config_fp, weight_feature_name, weight_feature_id, reverse, alpha, max_iter):
        UDG.__init__(self, config_fp, weight_feature_name, weight_feature_id, reverse)
        self.feature_name = '%s_%s_%s' % (self.feature_name, str(alpha), str(max_iter))
        self.pr = Graph.generate_pagerank(self.G, alpha, max_iter)

    def extract_row(self, row):
        q1 = str(row['question1']).strip()
        q2 = str(row['question2']).strip()

        qid1 = self.q2id[q1]
        qid2 = self.q2id[q2]
        pr1 = self.pr[qid1] * 1e6
        pr2 = self.pr[qid2] * 1e6
        return [pr1, pr2, max(pr1, pr2), min(pr1, pr2), (pr1 + pr2) / 2.]

    def get_feature_num(self):
        return 5


class GraphHits(UDG):
    def __init__(self, config_fp, weight_feature_name=None, weight_feature_id=None, reverse=False, max_iter=100):
        UDG.__init__(self, config_fp, weight_feature_name, weight_feature_id, reverse)
        self.hits_h, self.hits_a = Graph.generate_hits(self.G, max_iter)

    def extract_row(self, row):
        q1 = str(row['question1']).strip()
        q2 = str(row['question2']).strip()

        qid1 = self.q2id[q1]
        qid2 = self.q2id[q2]
        h1 = self.hits_h[qid1] * 1e6
        h2 = self.hits_h[qid2] * 1e6
        a1 = self.hits_a[qid1] * 1e6
        a2 = self.hits_a[qid2] * 1e6
        return [h1, h2, a1, a2,
                max(h1, h2), max(a1, a2),
                min(h1, h2), min(a1, a2),
                (h1 + h2) / 2., (a1 + a2) / 2.]

    def get_feature_num(self):
        return 10


class GraphShortestPath(UDG):

    def extract_row(self, row):
        q1 = str(row['question1']).strip()
        q2 = str(row['question2']).strip()
        qid1 = self.q2id[q1]
        qid2 = self.q2id[q2]
        shortest_path = -1
        self.G.remove_edge(qid1, qid2)
        if nx.has_path(Graph.G, qid1, qid2):
            shortest_path = nx.dijkstra_path_length(self.G, qid1, qid2)
        self.G.add_edge(qid1, qid2, weight=self.e2weight[(qid1, qid2)])
        return [shortest_path]

    def get_feature_num(self):
        return 1


class GraphNodeNeighborProperty(UDG):

    def extract_row(self, row):
        q1 = str(row['question1']).strip()
        q2 = str(row['question2']).strip()
        qid1 = self.q2id[q1]
        qid2 = self.q2id[q2]

        l = []
        r = []
        l_nb = self.G.neighbors(qid1)
        r_nb = self.G.neighbors(qid2)
        for n in l_nb:
            if (n != qid2) and (n != qid1):
                l.append(self.e2weight[(qid1, n)])
        for n in r_nb:
            if (n != qid2) and (n != qid1):
                r.append(self.e2weight[(qid2, n)])

        aggregation_modes = ["mean", "std", "max", "min", "median"]
        fs = MathUtil.aggregate(l, aggregation_modes) + MathUtil.aggregate(r, aggregation_modes)
        return fs

    def get_feature_num(self):
        return 10


class GraphNodeNeighborShareNum(UDG):

    def extract_row(self, row):
        q1 = str(row['question1']).strip()
        q2 = str(row['question2']).strip()
        qid1 = self.q2id[q1]
        qid2 = self.q2id[q2]
        l_nb = self.G.neighbors(qid1)
        r_nb = self.G.neighbors(qid2)
        fs = list()
        fs.append(len(list((set(l_nb).union(set(r_nb))) ^ (set(l_nb) ^ set(r_nb)))))
        return fs

    def get_feature_num(self):
        return 1


def demo():
    config_fp = '/Users/houjianpeng/Github/kaggle-quora-question-pairs/conf/featwheel.conf'

    GraphEdgeMaxCliqueSize(config_fp).extract('train')
    GraphEdgeMaxCliqueSize(config_fp).extract('test')


if __name__ == '__main__':
    demo()
