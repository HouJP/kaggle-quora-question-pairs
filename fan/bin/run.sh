
logfile='./log/randomforest.ne-200.gini.mf-07.md-6.mss-6.msl-3.mwfl-0.1e-7.rs-11.hjp.log'
nohup python model_randomforest.py ../conf/python.conf train > $logfile  & 2>&1
tail -f -n 100 $logfile
