#!/bin/bash

dir='data_utanet'

./seq2seq_vae.py -u 300 \
		 -t gru \
		 -l 2 \
		 -e 40 \
		 -b 10 \
		 --n_embed 300 \
		 --word_dropout 0.38 \
		 --denoising_rate 0.0 \
		 --n_latent 100 \
		 --validation-interval 2000 \
		 --log-interval 200 \
		 --max-source-sentence 50 \
		 --validation-source $dir/test.txt \
		 --validation-target $dir/test.txt \
		 $dir/train.txt $dir/train.txt \
		 $dir/train.vocab $dir/train.vocab

		 #--resume result_0213_20000/model_iter_134444.npz \
