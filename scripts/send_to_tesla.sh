#!/usr/bin/env bash

rsync -r -v -e \
	'ssh -p25252' \
	/home/joaocarvalho/workspace/colorization-network/ jcarvalho@tesla.dcc.ufrj.br:/home/jcarvalho/ \
	--exclude=imagenet/ --exclude=.git/ --exclude=weights/ --exclude=.idea/ \
	--exclude=graph/ --exclude=results/ --exclude=check.out --exclude=weights.npy \
	--exclude=tesla/ --exclude=*.out --exclude=tensorboard/ --exclude=*.tar \
	--exclude=data_256/ --exclude=imagenet/ --exclude=tesla/

