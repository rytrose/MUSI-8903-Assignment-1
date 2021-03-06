#!/bin/sh
#############################################################################
# TODO: Initialize anything you need for the forward pass
#############################################################################
python -u train.py \
    --model twolayernn \
    --hidden-dim 512 \
    --epochs 4 \
    --weight-decay 0.0 \
    --momentum 0.0 \
    --batch-size 512 \
    --lr 0.001 | tee twolayernn.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
