#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import six

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.cuda import to_cpu
from chainer import serializers

from numpy import random

UNK = 0
EOS = 1


def sequence_embed(embed, xs):
    x_len = [len(x) for x in xs]
    x_section = numpy.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    exs = F.split_axis(ex, x_section, 0)
    return exs


class Seq2seq(chainer.Chain):

    def __init__(self, n_layers, n_source_vocab, n_target_vocab,
                 n_embed, n_units, n_latent,
                 type_unit, word_dropout, denoising_rate):
        super(Seq2seq, self).__init__()
        with self.init_scope():
            self.embed_x = L.EmbedID(n_source_vocab, n_embed)
            self.encoder = L.NStepGRU(n_layers, n_embed, n_units, 0.5)
            self.W_mu = L.Linear(n_units * n_layers, n_latent)
            self.W_ln_var = L.Linear(n_units * n_layers, n_latent)

            self.W_h = L.Linear(n_latent, n_units * n_layers)
            self.decoder = L.NStepGRU(n_layers, n_embed, n_units, 0.5)
            self.W = L.Linear(n_units, n_target_vocab)
            self.embed_y = L.EmbedID(n_target_vocab, n_embed)

        self.n_layers = n_layers
        self.n_units = n_units
        self.n_embed = n_embed
        self.word_dropout = word_dropout
        self.denoising_rate = denoising_rate
        self.n_latent = n_latent
        self.C = 0
        self.k = 10 # unstable if 5
        self.n_target_vocab = n_target_vocab
        

    def __call__(self, xs, ys):
        eos = self.xp.array([EOS], 'i')
        
        xs = [self.denoiseInput(x[::-1], self.denoising_rate) for x in xs] # denoising

        #ys_d = [self.wordDropout(y, self.word_dropout) for y in ys] # word dropout
        ys_d = [self.denoiseInput(y, self.word_dropout) for y in ys] # word dropout
        ys_in = [F.concat([eos, y], axis=0) for y in ys_d]
        ys_out = [F.concat([y, eos], axis=0) for y in ys]

        # Both xs and ys_in are lists of arrays.
        exs = sequence_embed(self.embed_x, xs)
        eys = sequence_embed(self.embed_y, ys_in)

        batch = len(xs)
        # None represents a zero vector in an encoder.
        hx, at = self.encoder(None, exs) # layer x batch x n_units
        hx_t = F.transpose(hx, (1,0,2))  # batch x layer x n_units
        mu = self.W_mu(hx_t) # batch x n_latent
        ln_var = self.W_ln_var(hx_t)
        #print('{},{}'.format(mu.shape,ln_var.shape))
        #print(hx_t.shape)

        rec_loss = 0
        concat_ys_out = F.concat(ys_out, axis=0)
        for _ in range(self.k):
            z = F.gaussian(mu, ln_var)
            z_e = F.expand_dims(z, 2) # batch x n_latent x 1
            Wz = self.W_h(z_e)        # batch x (layer x unit) 
            #print('Wz: {}, {}'.format(Wz.shape, type(Wz)))
            hys = F.split_axis(Wz, self.n_layers, 1) # layer x batch x unit
            #print('hys, {}'.format([x.shape for x in hys]))
            c_hy = F.concat([F.expand_dims(hy,0) for hy in hys], 0) # layer x batch x unit
            #print('c_hy: {}'.format(c_hy.shape))
            _, os = self.decoder(c_hy, eys)
            #print(len(os))
            concat_os = F.concat(os, axis=0)
            rec_loss += F.sum(F.softmax_cross_entropy(
                self.W(concat_os), concat_ys_out, reduce='no')) / (self.k * batch)
        latent_loss = F.gaussian_kl_divergence(mu, ln_var) / batch
        loss = rec_loss + self.C * latent_loss

        # wy = self.W(concat_os)
        # ys = self.xp.argmax(wy.data, axis=1).astype('i')
        # print(ys)
        
        chainer.report({'loss': loss.data}, self)
        chainer.report({'rec': rec_loss.data}, self)
        chainer.report({'lat': latent_loss.data}, self)
        n_words = concat_ys_out.shape[0]
        perp = self.xp.exp(loss.data * batch / n_words)
        chainer.report({'perp': perp}, self)

        return loss
    
    
    def denoiseInput(self, y, rate): # Denoising AutoEncoder
        if rate > 0.0:
            unk = self.xp.array([UNK], 'i') # replace into UNK
            len_y = len(y)
            n_replace = int(len_y * rate)
            #print('replace {}/{}'.format(n_replace,len_y))
            idx_replace = random.choice(range(len_y), n_replace)
            for i in idx_replace:
                y[i] = unk                    
        return y

    
    def wordDropout(self, y, rate): # Word Dropout
        if rate > 0.0:
            for i in range(len(y)):
                if self.xp.random.rand() < rate:
                    noise = self.xp.random.randint(self.n_target_vocab)
                    y[i] = self.xp.array([noise], 'i') # replace into random word
        return y


    def decode(self, h, max_length):
        def avoid_unk(wy_data):
            ys_0 = self.xp.argmax(wy_data, axis=1).astype('i')
            # this is darty implementation
            for i in range(ys_0.shape[0]):
                if ys_0[i] == self.xp.array([UNK], 'i'):
                    wy_data[i, ys_0[i]] = 0.0
                    ys_0[i] = self.xp.argmax(wy_data[i,:]).astype('i')
            return ys_0

        batch = h.shape[1]
        #with chainer.no_backprop_mode(), chainer.using_config('train', False):
        ys = self.xp.full(batch, EOS, 'i')
        result = []
        for i in range(max_length): # 学習のときとは異なり、1語ずつ計算
            eys = self.embed_y(ys)
            eys = F.split_axis(eys, batch, 0)
            h, ys = self.decoder(h, eys)
            concat_ys = F.concat(ys, axis=0)
            wy = self.W(concat_ys)
            ys = avoid_unk(wy.data)
            #ys = self.xp.argmax(wy.data, axis=1).astype('i')
            result.append(ys)

        # Using `xp.concatenate(...)` instead of `xp.stack(result)` here to
        # support NumPy 1.9.
        result = cuda.to_cpu(
            self.xp.concatenate([self.xp.expand_dims(x, 0) for x in result]).T)

        # Remove EOS taggs
        outs = []
        for y in result:
            inds = numpy.argwhere(y == EOS)
            if len(inds) > 0:
                y = y[:inds[0, 0]]
            outs.append(y)
        return outs

    
    def translate(self, xs, max_length=100):
        batch = len(xs)
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            xs = [x[::-1] for x in xs]
            exs = sequence_embed(self.embed_x, xs)
            h, a = self.encoder(None, exs)
            
            h_t = F.transpose(h, (1,0,2))
            mu = self.W_mu(h_t)
            ln_var = self.W_ln_var(h_t)        
            z = F.gaussian(mu, ln_var)
            z_e = F.expand_dims(z, 2) # batch x n_latent x 1
            Wz = self.W_h(z_e)
            hys = F.split_axis(Wz, self.n_layers, 1)
            h = F.concat([F.expand_dims(hy,0) for hy in hys], 0)
            outs = self.decode(h, max_length)

        return outs


    def generate(self, batch_size, max_length=100):
        with chainer.no_backprop_mode(), chainer.using_config('train', False):        
            z = self.xp.random.normal(0.0, 1.0, (batch_size, self.n_latent))\
                            .astype('f')
            Wz = self.W_h(z)
            hys = F.split_axis(Wz, self.n_layers, 1)
            h = F.concat([F.expand_dims(hy,0) for hy in hys], 0)
            outs = self.decode(h, max_length)

        return outs

    
    def latent_vector(self, xs):
        batch = len(xs)
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            xs = [x[::-1] for x in xs]
            exs = sequence_embed(self.embed_x, xs)
            h, _ = self.encoder(None, exs)
            h_t = F.transpose(h, (1,0,2))
            mu = self.W_mu(h_t)
                
        latent_vecs = to_cpu(mu.data)    # to cpu
        return latent_vecs

    
    def generate_by_latent(self, latent_vecs, max_length=100):
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            Wz = self.W_h(latent_vecs)
            hys = F.split_axis(Wz, self.n_layers, 1)
            h = F.concat([F.expand_dims(hy,0) for hy in hys], 0)
            outs = self.decode(h, max_length)

        return outs
