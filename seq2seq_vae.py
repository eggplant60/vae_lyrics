#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import json

from nltk.translate import bleu_score
from rouge import Rouge
import numpy
import progressbar
import six

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

from chainer import serializers
from numpy import random

import sys
sys.path.append('/home/naoya/work/text/uta-net')
import read_db

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
            # if attr:
            #     self.Wc = L.Linear(2*n_units, n_units)

        self.n_layers = n_layers
        self.n_units = n_units
        self.n_embed = n_embed
        self.word_dropout = word_dropout
        self.denoising_rate = denoising_rate
        self.n_latent = n_latent
        self.C = 0
        self.k = 10
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
        #print(mu.shape)
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
            # this is darty
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
    
    
def convert(batch, device):
    def to_device_batch(batch):
        if device is None:
            return batch
        elif device < 0:
            return [chainer.dataset.to_device(device, x) for x in batch]
        else:
            xp = cuda.cupy.get_array_module(*batch)
            concat = xp.concatenate(batch, axis=0)
            sections = numpy.cumsum([len(x) for x in batch[:-1]], dtype='i')
            concat_dev = chainer.dataset.to_device(device, concat)
            batch_dev = cuda.cupy.split(concat_dev, sections)
            return batch_dev

    return {'xs': to_device_batch([x for x, _ in batch]),
            'ys': to_device_batch([y for _, y in batch])}


class CalculateBleu(chainer.training.Extension):

    trigger = 1, 'epoch'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(
            self, model, test_data, key, batch=50, device=-1, max_length=100):
        self.model = model
        self.test_data = test_data
        self.key = key
        self.batch = batch
        self.device = device
        self.max_length = max_length

    def __call__(self, trainer):
        with chainer.no_backprop_mode():
            references = []
            hypotheses = []
            for i in range(0, len(self.test_data), self.batch):
                sources, targets = zip(*self.test_data[i:i + self.batch])
                references.extend([[t.tolist()] for t in targets])

                sources = [
                    chainer.dataset.to_device(self.device, x) for x in sources]
                ys = [y.tolist()
                      for y in self.model.translate(sources, self.max_length)]
                hypotheses.extend(ys)

        bleu = bleu_score.corpus_bleu(
            references, hypotheses,
            smoothing_function=bleu_score.SmoothingFunction().method1)
        chainer.report({self.key: bleu})

        
class CalculateRouge(chainer.training.Extension):

    trigger = 1, 'epoch'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(
            self, model, test_data, key, batch=50, device=-1, max_length=100):
        self.model = model
        self.test_data = test_data
        self.key = key
        self.batch = batch
        self.device = device
        self.max_length = max_length
        self.rouge = Rouge()

    def __call__(self, trainer):
        with chainer.no_backprop_mode():
            references = []
            hypotheses = []
            for i in range(0, len(self.test_data), self.batch):
                sources, targets = zip(*self.test_data[i:i + self.batch])
                references.extend([' '.join(map(str, t.tolist())) for t in targets])

                sources = [
                    chainer.dataset.to_device(self.device, x) for x in sources]
                ys = [' '.join(map(str, y.tolist()))
                      for y in self.model.translate(sources, self.max_length)]
                hypotheses.extend(ys)

        scores = self.rouge.get_scores(hypotheses, references, avg=True)
        rouge_l = scores["rouge-l"]
        chainer.report({self.key[0]: rouge_l["p"]})
        chainer.report({self.key[1]: rouge_l["r"]})
        chainer.report({self.key[2]: rouge_l["f"]})

        
def count_lines(path):
    with open(path) as f:
        return sum([1 for _ in f])


def load_vocabulary(path):
    with open(path) as f:
        # +2 for UNK and EOS
        word_ids = {line.strip(): i + 2 for i, line in enumerate(f)}
    word_ids['<UNK>'] = 0
    word_ids['<EOS>'] = 1
    return word_ids


def load_data(vocabulary, path):
    n_lines = count_lines(path)
    bar = progressbar.ProgressBar()
    data = []
    print('loading...: %s' % path)
    with open(path) as f:
        for line in bar(f, max_value=n_lines):
            words = line.strip().split()
            array = numpy.array([vocabulary.get(w, UNK) for w in words], 'i')
            data.append(array)
    return data


def calculate_unknown_ratio(data):
    unknown = sum((s == UNK).sum() for s in data)
    total = sum(s.size for s in data)
    return unknown / total


def main():
    parser = argparse.ArgumentParser(description='Chainer example: seq2seq')
    parser.add_argument('SOURCE', help='source sentence list')
    parser.add_argument('TARGET', help='target sentence list')
    parser.add_argument('SOURCE_VOCAB', help='source vocabulary file')
    parser.add_argument('TARGET_VOCAB', help='target vocabulary file')
    parser.add_argument('--validation-source',
                        help='source sentence list for validation')
    parser.add_argument('--validation-target',
                        help='target sentence list for validation')
    parser.add_argument('--batchsize', '-b', type=int, default=10,
                        help='number of sentence pairs in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=50,
                        help='number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--resume', '-r', default='',
                        help='resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1024,
                        help='number of units')
    parser.add_argument('--type_unit', '-t', choices={'lstm', 'gru'},
                        help='number of units')
    parser.add_argument('--layer', '-l', type=int, default=3,
                        help='number of layers')
    parser.add_argument('--min-source-sentence', type=int, default=2, # for caluculation of ngram 2
                        help='minimium length of source sentence')
    parser.add_argument('--max-source-sentence', type=int, default=500,
                        help='maximum length of source sentence')
    parser.add_argument('--min-target-sentence', type=int, default=2, # for caluculation of ngram 2
                        help='minimium length of target sentence')
    parser.add_argument('--max-target-sentence', type=int, default=50,
                        help='maximum length of target sentence')
    parser.add_argument('--log-interval', type=int, default=200,
                        help='number of iteration to show log')
    parser.add_argument('--validation-interval', type=int, default=1000,
                        help='number of iteration to evlauate the model '
                        'with validation dataset')
    parser.add_argument('--word_dropout', '-w', type=float, default=0.0)
    parser.add_argument('--denoising_rate', '-d', type=float, default=0.0)
    parser.add_argument('--n_latent', type=int, default=100)
    parser.add_argument('--n_embed', type=int, default=512,
                        help='length of embedding')
        
    args = parser.parse_args()


    source_ids = load_vocabulary(args.SOURCE_VOCAB)
    target_ids = load_vocabulary(args.TARGET_VOCAB)
    train_source = load_data(source_ids, args.SOURCE)
    train_target = load_data(target_ids, args.TARGET)
    assert len(train_source) == len(train_target)
    train_data = [(s, t)
                  for s, t in six.moves.zip(train_source, train_target)
                  if args.min_source_sentence <= len(s)
                  <= args.max_source_sentence and
                  args.min_source_sentence <= len(t)
                  <= args.max_source_sentence]
    train_source_unknown = calculate_unknown_ratio(
        [s for s, _ in train_data])
    train_target_unknown = calculate_unknown_ratio(
        [t for _, t in train_data])

    print('Source vocabulary size: %d' % len(source_ids))
    print('Target vocabulary size: %d' % len(target_ids))
    print('Train data size: %d' % len(train_data))
    print('Train source unknown ratio: %.2f%%' % (train_source_unknown * 100))
    print('Train target unknown ratio: %.2f%%' % (train_target_unknown * 100))

    target_words = {i: w for w, i in target_ids.items()}
    source_words = {i: w for w, i in source_ids.items()}


    model = Seq2seq(args.layer, len(source_ids), len(target_ids),
                    args.unit, args.n_embed, args.n_latent,
                    args.type_unit, args.word_dropout, args.denoising_rate)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)
    updater = training.StandardUpdater(
        train_iter, optimizer, converter=convert, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'))
    trainer.extend(extensions.LogReport(
        trigger=(args.log_interval, 'iteration')))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration',
         'main/loss', 'main/rec', 'main/lat', 'main/perp',
         'bleu', 'p', 'r', 'f', 'elapsed_time']),
        trigger=(args.log_interval, 'iteration'))
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}.npz'),
        trigger=(5, 'epoch'))

    if args.validation_source and args.validation_target:
        test_source = load_data(source_ids, args.validation_source)
        test_target = load_data(target_ids, args.validation_target)
        assert len(test_source) == len(test_target)
        test_data = list(six.moves.zip(test_source, test_target))
        test_data = [(s, t) for s, t in test_data if 0 < len(s) and 0 < len(t)]
        test_source_unknown = calculate_unknown_ratio(
            [s for s, _ in test_data])
        test_target_unknown = calculate_unknown_ratio(
            [t for _, t in test_data])

        print('Validation data: %d' % len(test_data))
        print('Validation source unknown ratio: %.2f%%' %
              (test_source_unknown * 100))
        print('Validation target unknown ratio: %.2f%%' %
              (test_target_unknown * 100))

        # @chainer.training.make_extension()
        # def translate(trainer):
        #     source, target = test_data[numpy.random.choice(len(test_data))]
        #     result = model.translate([model.xp.array(source)])[0]

        #     source_sentence = ' '.join([source_words[x] for x in source])
        #     target_sentence = ' '.join([target_words[y] for y in target])
        #     result_sentence = ' '.join([target_words[y] for y in result])
        #     #print('#  source : ' + source_sentence)
        #     print('#  result : ' + result_sentence)
        #     print('#  expect : ' + target_sentence)

        # trainer.extend(
        #     translate, trigger=(args.validation_interval, 'iteration'))
        
        @chainer.training.make_extension()
        def generate(trainer):
            results = model.generate(5)
            for i, result in enumerate(results):
                print('#  result {}: {}'.format(i+1, ' '.join([source_words[x] for x in result])))
                
        trainer.extend(
            generate, trigger=(args.validation_interval, 'iteration'))
        
        trainer.extend(
            CalculateBleu(
                model, test_data, 'bleu', device=args.gpu),
            trigger=(args.validation_interval, 'iteration'))
        trainer.extend(
            CalculateRouge(
                model, test_data, ['p', 'r', 'f'], device=args.gpu),
            trigger=(args.validation_interval, 'iteration'))
        

    @chainer.training.make_extension()
    def fit_C(trainer):
        # if updater.epoch < 10:
        #     model.C = 0.0
        # else:
        #     model.C = 0.06 * (updater.epoch - 10) / args.epoch
        if model.C < 0.5 and updater.epoch > 10:
            model.C += 0.008
        print('epoch: {}, C: {},'.format(updater.epoch, model.C))

    trainer.extend(fit_C, trigger=(1, 'epoch'))

        
    if args.resume:
        serializers.load_npz(args.resume, model)
    
    print('start training')
    trainer.run()
    print('complete training')

    with open('result/args.txt', 'w') as f:
        args_dict = {}
        for i in dir(args):
            if '_' in i[0]: continue
            args_dict[str(i)] = getattr(args, i)
        json.dump(args_dict, f, ensure_ascii=False,
                  indent=4, sort_keys=True, separators=(',', ': '))
        
    serializers.save_npz('result/model.npz', model)
    
if __name__ == '__main__':
    main()
