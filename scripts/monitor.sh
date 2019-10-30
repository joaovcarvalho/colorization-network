#!/usr/bin/env bash

ssh -t jcarvalho@tesla.dcc.ufrj.br -p 25252 'tail -f imagenet_train.out'
#ssh -t jcarvalho@tesla.dcc.ufrj.br -p 25252 'tail -f check.out'
