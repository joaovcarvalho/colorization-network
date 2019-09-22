#!/usr/bin/env bash

scp -P 25252 -r jcarvalho@tesla.dcc.ufrj.br:weights/ ./
scp -P 25252 -r jcarvalho@tesla.dcc.ufrj.br:weights.npy weights.npy
