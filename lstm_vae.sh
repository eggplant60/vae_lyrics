#!/bin/bash

dir='data_utanet'

./seq2seq_vae.py -u 200 \
		 -t gru \
		 -l 2 \
		 -e 40 \
		 -b 10 \
		 --n_embed 300 \
		 --word_dropout 0.38 \
		 --denoising_rate 0.0 \
		 --n_latent 100 \
		 --validation-interval 2000 \
		 --max-source-sentence 50 \
		 --validation-source $dir/test.txt \
		 --validation-target $dir/test.txt \
		 $dir/train.txt $dir/train.txt \
		 $dir/train.vocab $dir/train.vocab
#	     --validation-source $dir/train_title.txt.mini \
#	     --validation-target $dir/train_title.txt.mini \
