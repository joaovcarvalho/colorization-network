#!/usr/bin/env bash
sshpass -p '$@Jv.21301410' scp -P 25252 -r jcarvalho@tesla.dcc.ufrj.br:imagenet_train.out imagenet_train.out
tail imagenet_train.out
