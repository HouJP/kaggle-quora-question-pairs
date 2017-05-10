#########################################################################
# File Name: download.sh
# Author: HouJP
# mail: houjp1992@gmail.com
# Created Time: 日  3/26 19:57:26 2017
#########################################################################
#! /bin/bash

function run() {
	user=$1
	address=$2
    tag=$3
	
	server_project_pt="/home/${user}/kaggle-quora-question-pairs/"
	local_project_pt="//Users/houjianpeng/Github/kaggle-quora-question-pairs/"

    # create directory
    mkdir $local_project_pt/data/out/$tag
#    mkdir $local_project_pt/data/out/$tag/pred/
#    mkdir $local_project_pt/data/out/$tag/model/
#    mkdir $local_project_pt/data/out/$tag/conf/
    # download
    scp -r ${user}@${address}:$server_project_pt/data/out/$tag/conf/ $local_project_pt/data/out/$tag/
    scp -r ${user}@${address}:$server_project_pt/data/out/$tag/model/ $local_project_pt/data/out/$tag/
    scp -r ${user}@${address}:$server_project_pt/data/out/$tag/pred/ $local_project_pt/data/out/$tag/
    # zip
#    zip -r $local_project_pt/data/out/$tag/pred/$score.zip $local_project_pt/data/out/$tag/pred/full.test.pred
    # open
    open $local_project_pt/data/out/$tag/pred/
}

if [ $# -ne 3 ]; then
    echo "Usage: download <user> <address> <tag>"
    exit 255
fi

user=$1
address=$2
tag=$3

run $user $address $tag
