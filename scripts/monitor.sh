#!/usr/bin/env bash

FILE_NAME='imagenet_train.out'
#FILE_NAME='check.out'

# watch "sshpass -p '$@Jv.21301410' scp -P 25252 -r jcarvalho@tesla.dcc.ufrj.br:$FILE_NAME $FILE_NAME && tail $FILE_NAME"
ssh -t jcarvalho@tesla.dcc.ufrj.br -p 25252 'tail -f imagenet_train.out'
