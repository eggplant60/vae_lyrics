#!/bin/bash

PY='./wmt_preprocess.py'
VOCAB_SIZE='45000'

$PY $1/train_l.txt $1/train.txt --vocab-file $1/train.vocab \
    --vocab-size $VOCAB_SIZE
$PY $1/test_l.txt $1/test.txt
cat $1/test.txt | awk '{print NF}' > $1/test_cnt.txt

