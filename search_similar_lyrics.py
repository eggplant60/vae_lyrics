#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

from net import Seq2seq
from extract_vector_vae import load_model_vocab
from lstm_vae import load_data
from scipy.spatial.distance import euclidean, cosine

import numpy as np
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='search lyrics similar to the query')
    parser.add_argument('--result_dir', '-r', type=str, default='result_0209_C024')
    parser.add_argument('--query', '-q', type=str, required=True)
    args = parser.parse_args()
    
    source_ids, _, \
    train_source, _, \
    test_source, _, model = load_model_vocab(args.result_dir)

    query_source = load_data(source_ids, args.query)[:100]
    xs = [ model.xp.array(source) for source in query_source ]
    qs = model.latent_vector(xs)

    vs = []
    for i in range(0,len(train_source),50):
        xs = [ model.xp.array(source) for source in train_source[i:i+50] ]
        vs.extend(model.latent_vector(xs))
    print(len(vs))

    source_words = {i: w for w, i in source_ids.items()}
    with open(args.query, 'r') as f:
        query_txt = f.readlines()


    for i in range(len(query_source)):

        print('-' * 80)
        print('query {}'.format(i+1))
        print('{}'.format(query_txt[i].replace(' / ', '\n')))

        similarity = np.array([cosine(qs[i,:], v) for v in vs])
        #similarity = numpy.array([euclidean(qs[i,:], v) for v in vs])
        similar_idx = np.argsort(similarity)[:3] # 距離の近い順

        for num, idx in enumerate(similar_idx):
            decode_string = ''.join([source_words[x]
                                     for x in train_source[idx]]).replace('/','\n')
            print('No. {}: similarity: {}, {} words'.\
                  format(num+1, similarity[idx], len(train_source[idx])))
            print(decode_string)
            print()

    
