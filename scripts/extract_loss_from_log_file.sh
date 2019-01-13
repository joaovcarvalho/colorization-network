#!/usr/bin/env bash
cat imagenet_train.out | grep loss | awk '{print $8}' > loss.out
