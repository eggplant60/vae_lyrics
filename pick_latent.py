#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pylab
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.decomposition import PCA

from util_latent import test_data


def plot_event(df, x, y, hue):

    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(5,10))

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
            print('count: {}'.format(df['count'][i]))
            print('')
            ax2.plot([1], [df['count'][i]], 'o')
        fig.canvas.draw() # re-draw


    #df_sorted = df.sort_values(by=[hue], ascending=True, inplace=False)
    #df_sorted.reset_index(drop=True)
    
    hue_unique = sorted(list(df[hue].unique())) # in ascending order
    def get_index(hue_value):
        return hue_unique.index(hue_value)
    
    pred_color = np.array([get_index(v) for v in df[hue].__iter__()]) / len(hue_unique)

    ax1.scatter(df[x], df[y], c=pred_color, cmap=cm.hsv, picker=3)
    ax1.set_xlabel(x)
    ax1.set_ylabel(y)
    ax1.set_title('Distribution')
    
    cid = fig.canvas.mpl_connect('pick_event', on_pick)

    plt.show()



if __name__ == '__main__':
    df = test_data('result_0209_C024/vector.pkl', 'data_utanet')
    vecs = np.array(list(df['vector']))

    # PCA
    pca = PCA(n_components=10)
    pca_latent = pca.fit_transform(vecs)
    df['pca1'] = pca_latent[:,0]
    df['pca2'] = pca_latent[:,1]

    # Plot
    plot_event(df, x='pca1', y='pca2', hue='artist')

