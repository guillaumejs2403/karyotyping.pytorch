#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

ONLYCHRM='True'
LR=0.0001
EPOCH=200
BATCH_SIZE=4
DECAY='False'
DIVIDER=100
SAVEDIR='/media/user_home4/gjeanneret/karyotyping.pytorch/psp_net-resnet34'
EACH=45
N=2
L=0.001



if [ $ONLYCHRM='True' ]
then
	python training_nocat.py --learning_rate $LR --epochs $EPOCH --batch_size $BATCH_SIZE --decay $DECAY --divider $DIVIDER --save_dir $SAVEDIR --each_ckpt $EACH --schedule 50 100 --Ln-n $N --Ln-lambda $L  2> errorlog_nocat.txt
else
	python training.py --learning_rate $LR --epochs $EPOCH --batch_size $BATCH_SIZE --decay $DECAY --divider $DIVIDER --save_dir $SAVEDIR --each_ckpt $EACH --schedule 100 --Ln-n $N --Ln-lambda $L 2> errorlog.txt
fi
