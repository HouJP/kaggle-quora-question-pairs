#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/6/22 14:30
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


import math


class Evaluator(object):

    valid_evaluator_name = ['entropy_loss']

    @staticmethod
    def evaluate(evaluator_name, labels, preds):
        assert evaluator_name.lower() in Evaluator.valid_evaluator_name, 'Wrong evaluator_name(%s)' % evaluator_name
        evaluator = getattr(Evaluator, evaluator_name)
        return evaluator(labels, preds)

    @staticmethod
    def entropy_loss(labels, preds):
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
        return s
