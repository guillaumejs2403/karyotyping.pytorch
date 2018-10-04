#!/bin/bash

clear
export CUDA_VISIBLE_DEVICES=2

CHRM='All'
LR=0.00001
EPOCH=300
BATCH_SIZE=8
DECAY='False'
DIVIDER=100
SAVEDIR='/media/user_home4/gjeanneret/karyotyping.pytorch/fcn8-pretrained'
EACH=50
N=2
L=0.001
MODEL='/media/user_home4/gjeanneret/karyotyping.pytorch/psp_net-resnet18/BGvsCHR_epochs_150_lr_0.0005_batchSize_4_L2_0.001/model.pkl'

echo $CHRM

if [ $CHRM = 'Onlychrm' ]; then
	echo "Background vs chromosomes"
	python training_nocat.py --learning_rate $LR --epochs $EPOCH --batch_size $BATCH_SIZE --decay $DECAY --divider $DIVIDER --save_dir $SAVEDIR --each_ckpt $EACH --schedule 50 100 --Ln-n $N --Ln-lambda $L  2> errorlog_nocat.txt
elif [ $CHRM = 'All' ]; then
	echo "All categories"
	python training.py --learning_rate $LR --epochs $EPOCH --batch_size $BATCH_SIZE --decay $DECAY --divider $DIVIDER --save_dir $SAVEDIR --each_ckpt $EACH --schedule 100 --Ln-n $N --Ln-lambda $L 2> errorlog.txt
else
	echo "Attention"
	python training_withattention.py --learning_rate $LR --epochs $EPOCH --batch_size $BATCH_SIZE --decay $DECAY --divider $DIVIDER --save_dir $SAVEDIR --each_ckpt $EACH --Ln-n $N --Ln-lambda $L --model_folder $MODEL 2>error_att.txt
fi
