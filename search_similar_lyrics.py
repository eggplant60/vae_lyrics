#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pickle
import os
from pprint import pprint

from chainer.cuda import to_cpu

from seq2seq_vae import *
from extract_vector_vae import *
import scipy.spatial.distance as dis


class Seq2seq_ride(Seq2seq):

    def out_vector(self, xs):
        batch = len(xs)
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            xs = [x[::-1] for x in xs]
            exs = sequence_embed(self.embed_x, xs)
            h, _ = self.encoder(None, exs)
            h_t = F.transpose(h, (1,0,2))
            mu = self.W_mu(h_t)
                
        #c_vectors = F.concat(vectors, axis=1) # layer
        #l_vectors = to_cpu(c_vectors.data)    # to cpu
        l_vectors = to_cpu(mu.data)    # to cpu
        return l_vectors


    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='search lyrics similar to the query')
    parser.add_argument('--result_dir', '-r', type=str, default='result_0208_C041')
    parser.add_argument('--query', '-q', type=str, required=True)
    args = parser.parse_args()
    
    source_ids, _, \
    train_source, _, \
    test_source, _, model = load_model_vocab(args.result_dir)

    query_source = load_data(source_ids, args.query)
    xs = [ model.xp.array(source) for source in query_source ]
    q = model.out_vector(xs)

    vs = []
    for i in range(0,len(train_source),50):
        xs = [ model.xp.array(source) for source in train_source[i:i+50] ]
        vs.extend(model.out_vector(xs))
    print(len(vs))


    similarity = numpy.array([dis.cosine(q, v) for v in vs])
    similar_idx = numpy.argsort(similarity)[:5] # 距離の近い順

    source_words = {i: w for w, i in source_ids.items()}

    with open(args.query, 'r') as f:
        print('query: {}'.format(f.readlines()))
    
    for i, idx in enumerate(similar_idx):
        decode_string = ''.join([source_words[x]
                                 for x in train_source[idx]]).replace('/','\n')
        print('No. {}: similarity: {}, {} words'.\
              format(i+1, similarity[idx], len(train_source[idx])))
        print(decode_string)
        print()

    
