#########################################################################
# File Name: download.sh
# Author: HouJP
# mail: houjp1992@gmail.com
# Created Time: 日  3/26 19:57:26 2017
#########################################################################
#! /bin/bash

server_project_pt="/home/fyx/kaggle-quora-question-pairs/"
local_project_pt="//Users/houjianpeng/Github/kaggle-quora-question-pairs/"

function run() {
    tag=$1
    score=$2
    # create directory
    mkdir $local_project_pt/data/out/$tag
    mkdir $local_project_pt/data/out/$tag/pred/
    # download
    scp -r fyx@kaggle:$server_project_pt/data/out/$tag/pred/full.test.pred $local_project_pt/data/out/$tag/pred/
    # zip
    zip -r $local_project_pt/data/out/$tag/pred/$score.zip $local_project_pt/data/out/$tag/pred/full.test.pred
}

if [ $# -ne 2 ]; then
    echo "Usage: download <tag> <score>"
    exit 255
fi

tag=$1
score=$2

run $tag $score