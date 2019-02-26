#!/usr/bin/env bash
cat imagenet_train.out | grep loss | grep -v home | awk '{print $8}' > loss.out
