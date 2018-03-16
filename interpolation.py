#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

from lstm_vae import load_data
from extract_vector_vae import load_model_vocab
from scipy.spatial.distance import euclidean, cosine

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='interpolate between given two query')
    parser.add_argument('--result_dir', '-r', type=str, default='result_0209_C024')
    parser.add_argument('--query', '-q', type=str, required=True)
    args = parser.parse_args()
    
    source_ids, _, \
    train_source, _, \
    test_source, _, model = load_model_vocab(args.result_dir)

    source_words = {i: w for w, i in source_ids.items()}

    # extract latent vectors with query texts.
    query_source = load_data(source_ids, args.query)[:2]
    xs = [ model.xp.array(source) for source in query_source ]
    h1, h2 = model.latent_vector(xs)
    
    # print two query texts.
    with open(args.query, 'r') as f:
        for i, line in enumerate(f.readlines()[:2]):
            print('query {}'.format(i+1))
            print(line.replace(' / ', '\n'))

    for i in range(6):
        r = i * 0.2
        h = (1.0-r) * h1 + r * h2
        h = h[model.xp.newaxis, :]
        h = model.xp.array(h)
        results = model.generate_by_latent(h)
        for result in results: # 1回のみ
            decode_string = ''.join([source_words[x] for x in result]).replace('/','\n')
            print('h1 : h2 = {:0.2f} : {:0.2f}'.format(1.0-r, r))
            print(decode_string)
            print()

