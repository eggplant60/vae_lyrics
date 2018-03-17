#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pylab
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from util_latent import test_data

def on_pick(event):
    # line = event.artist
    # xdata, ydata = line.get_data()
    indexes = event.ind
    print('on pick')
    for i in indexes:
        print('index: {}'.format(i))
        print('artist: {}'.format(df['artist'][i]))
        print('title: {}'.format(df['title'][i]))
        print('lyric: {}'.format(df['lyric'][i]))
        print('')
    
df = test_data('result_0209_C024/vector.pkl', 'data_utanet')
vecs = np.array(list(df['vector']))

# PCA
pca = PCA(n_components=10)
pca_latent = pca.fit_transform(vecs)
df['pca_vec'] = pca_latent[:,0]

fig, ax = plt.subplots()
#ax.plot(vecs[:,0], vecs[:,1], 'o', picker=3)
ax.plot(df['count'],df['pca_vec'], 'o', picker=3)
cid = fig.canvas.mpl_connect('pick_event', on_pick)

# sns.pairplot(df)

plt.show()
