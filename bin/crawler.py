#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/6/15 23:32
# @Author  : HouJP
# @Email   : houjp1992@gmail.com

import cookielib
import urllib
import urllib2

from bin.featwheel.utils import TimeUtil


class LeaderBoard(object):

    def __init__(self, top_url, all_url, lb_pt):
        self.top_url = top_url
        self.all_url = all_url
        self.lb_pt = lb_pt

        self.operate = ''
        self.cj = cookielib.CookieJar()
        self.opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(self.cj))
        urllib2.install_opener(self.opener)

    def download_rank(self):
        self.operate = self._get_response(self.all_url)
        web_content = self.operate.read()
        self._save_data(web_content)

    def _save_data(self, data):
        f = open('%s/lb_%s.txt' % (self.lb_pt, TimeUtil.t_now_YmdH()), 'w')
        f.write('%s\n' % data)
        f.close()

    def _get_response(self, url, data=None):
        if data is not None:
            req = urllib2.Request(url, urllib.urlencode(data))
        else:
            req = urllib2.Request(url)

        response = self.opener.open(req)
        return response

if __name__ == "__main__":
    top_url = 'https://www.kaggle.com/c/6277/leaderboard.json?includeBeforeUser=false&includeAfterUser=false'
    all_url = 'https://www.kaggle.com/c/6277/leaderboard.json?includeBeforeUser=true&includeAfterUser=false'
    lb_pt = '/home/houjianpeng/kaggle-quora-question-pairs/data/leaderboard/'

    lb = LeaderBoard(top_url=top_url, all_url=all_url, lb_pt=lb_pt)
    lb.download_rank()
