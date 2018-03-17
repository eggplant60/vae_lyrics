#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pickle
from os.path import join


def test_data(pkl_file, data_dir):
    art_file = join(data_dir, 'test_a.txt')
    ttl_file = join(data_dir, 'test_t.txt')
    lyr_file = join(data_dir, 'test.txt')
    cnt_file = join(data_dir, 'test_cnt.txt')

    # artist file の情報
    with open(art_file, 'r') as f:
        art_list = [line.strip() for line in f.readlines()]
        
    # title file の情報
    with open(ttl_file, 'r') as f:
        ttl_list = [line.strip() for line in f.readlines()]

    # lyric file の情報
    with open(lyr_file, 'r') as f:
        lyr_list = [line.strip() for line in f.readlines()]
        
    # count file の情報
    with open(cnt_file, 'r') as f:
        cnt_list = [int(line) for line in f.readlines()]

    # pkl file の情報
    with open(pkl_file, 'rb') as f:
        vectors = [vec for vec in pickle.load(f)]

    df = pd.DataFrame({'artist': art_list,
                       'title' : ttl_list,
                       'lyric' : lyr_list,
                       'count' : cnt_list,
                       'vector': vectors,
    })
    return df
    


if __name__ == '__main__':
    df = test_data('result_0209_C024/vector.pkl', 'data_utanet')
    print('データセットの数: {}, 特徴量の次元: {}'.\
          format(len(df.index), df['vector'][0].shape))
    print('出現頻度')
    print(df['artist'].value_counts())
    #print(df)
