#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pickle
import os
from pprint import pprint

from chainer.cuda import to_cpu

from seq2seq_vae import *


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


    
def load_model_vocab(result_dir, test_data=None):

    with open(os.path.join(result_dir, 'args.txt'), 'r') as f:
        args_i = json.load(f)

    if test_data: # over write
        args_i['validation_source'] = test_data
        args_i['validation_target'] = test_data

    args_i['resume'] = os.path.join(result_dir, 'model.npz')
        
    pprint(args_i)
        
    source_ids = load_vocabulary(args_i['SOURCE_VOCAB'])
    target_ids = load_vocabulary(args_i['TARGET_VOCAB'])

    model = Seq2seq_ride(args_i['layer'], len(source_ids), len(target_ids),
                         args_i['unit'], args_i['n_embed'], args_i['n_latent'],
                         args_i['type_unit'], args_i['word_dropout'],
                         args_i['denoising_rate'])
    
    if args_i['gpu'] >= 0:
        chainer.cuda.get_device(args_i['gpu']).use()
        model.to_gpu(args_i['gpu'])
        
    if args_i['resume']:
        serializers.load_npz(args_i['resume'], model)

        
    train_source = load_data(source_ids, args_i['SOURCE'])
    train_target = load_data(target_ids, args_i['TARGET'])
    test_source = load_data(source_ids, args_i['validation_source'])
    test_target = load_data(target_ids, args_i['validation_target'])
    # assert len(test_source) == len(test_target)
    # test_data = list(six.moves.zip(test_source, test_target))
    # test_data = [(s, t) for s, t in test_data if 0 < len(s) and 0 < len(t)]
    # test_source_unknown = calculate_unknown_ratio(
    #     [s for s, _ in test_data])
    # test_target_unknown = calculate_unknown_ratio(
    #     [t for _, t in test_data])

    # print('Validation data: %d' % len(test_data))
    # print('Validation source unknown ratio: %.2f%%' %
    #       (test_source_unknown * 100))
    # print('Validation target unknown ratio: %.2f%%' %
    #       (test_target_unknown * 100))

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

    # save vector    
    xs = [ model.xp.array(source) for source in test_source[0:1000] ]
    vector = model.out_vector(xs)
    print(vector)

    pkl_file = os.path.join(args.RESULT_DIR, 'vector.pkl')
    
    with open(pkl_file, 'wb') as f:
        pickle.dump(vector, f)

    print('dump on \'{}\''.format(pkl_file))

