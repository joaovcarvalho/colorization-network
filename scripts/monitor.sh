#!/usr/bin/env bash

FILE_NAME='imagenet_train.out'
#FILE_NAME='check.out'

while :
do
    printf '\n Running... \n'
    scp -P 25252 -r jcarvalho@tesla.dcc.ufrj.br:$FILE_NAME $FILE_NAME
    tail $FILE_NAME
	sleep 2
done
