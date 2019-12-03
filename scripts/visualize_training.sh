#!/usr/bin/env bash

FILE_NAME='imagenet_train.out'

scp -P 25252 -r jcarvalho@tesla.dcc.ufrj.br:${FILE_NAME} ${FILE_NAME}
#cat ${FILE_NAME} | grep loss | grep -v home | awk '{print $9}' > loss.out
source activate tesla
python plot_loss.py