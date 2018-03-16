#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pickle
import os
from pprint import pprint
import json

import chainer
from chainer import serializers

import numpy as np

from net import Seq2seq
from lstm_vae import load_vocabulary, load_data

    
def load_model_vocab(result_dir, test_data=None, gpu=0):

    with open(os.path.join(result_dir, 'args.txt'), 'r') as f:
        args_i = json.load(f)

    if test_data: # over write
        args_i['validation_source'] = test_data
        args_i['validation_target'] = test_data

    args_i['resume'] = os.path.join(result_dir, 'model.npz')
        
    pprint(args_i)
        
    source_ids = load_vocabulary(args_i['SOURCE_VOCAB'])
    target_ids = load_vocabulary(args_i['TARGET_VOCAB'])

    model = Seq2seq(args_i['layer'], len(source_ids), len(target_ids),
                    args_i['unit'], args_i['n_embed'], args_i['n_latent'],
                    args_i['type_unit'], args_i['word_dropout'],
                    args_i['denoising_rate'])
    
    if gpu >= 0:
        chainer.cuda.get_device(gpu).use()
        model.to_gpu(gpu)
        
    if args_i['resume']:
        serializers.load_npz(args_i['resume'], model)
        
    train_source = load_data(source_ids, args_i['SOURCE'])
    train_target = load_data(target_ids, args_i['TARGET'])
    test_source = load_data(source_ids, args_i['validation_source'])
    test_target = load_data(target_ids, args_i['validation_target'])

    return source_ids, target_ids, \
        train_source, train_target, \
        test_source, test_target, \
        model

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract latent vectors')
    parser.add_argument('RESULT_DIR', type=str)
    parser.add_argument('--test_data', type=str, default=None)

    args = parser.parse_args()
    
    _, _, _, _, \
    test_source, _, \
    model = load_model_vocab(args.RESULT_DIR,
                             test_data=args.test_data)
    n_source = len(test_source)

    # save latent vectors
    latent_vecs = []
    batchsize = 50
    for i in range(0, n_source, batchsize):
        xs = [ model.xp.array(source)
               for source in test_source[i:min(i+batchsize, n_source)] ]
        vec = model.latent_vector(xs)
        latent_vecs.append(vec)
        #print(vector)
        
    latent_vecs = np.concatenate(latent_vecs, axis=0) # list -> numpy.ndarray
    pkl_file = os.path.join(args.RESULT_DIR, 'vector.pkl')
    
    with open(pkl_file, 'wb') as f:
        pickle.dump(latent_vecs, f)

    print('dump on \'{}\''.format(pkl_file))

