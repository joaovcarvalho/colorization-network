#!/usr/bin/env bash

FILE_NAME='imagenet_train.out'

scp -P 25252 -r jcarvalho@tesla.dcc.ufrj.br:${FILE_NAME} ${FILE_NAME}