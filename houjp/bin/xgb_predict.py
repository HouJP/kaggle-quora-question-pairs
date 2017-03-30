# -*- coding: utf-8 -*-
# ! /usr/bin/python

from model import Model
import sys

if __name__ == "__main__":
    if 2 != len(sys.argv):
        print "Usage: xgb_predict <tag>"
        exit(-1)

    tag = sys.argv[1]
    Model.run_predict_xgb(tag)