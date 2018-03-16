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


from net import Seq2seq, UNK, EOS

# import sys
# sys.path.append('/home/naoya/work/text/uta-net')
# import read_db

    
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


class CalculateBleuRouge(chainer.training.Extension):

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
            references_r = []
            hypotheses = []
            for i in range(0, len(self.test_data), self.batch):
                sources, targets = zip(*self.test_data[i:i + self.batch])
                references.extend([[t.tolist()] for t in targets])
                references_r.extend([' '.join(map(str, t.tolist())) for t in targets])

                sources = [
                    chainer.dataset.to_device(self.device, x) for x in sources]
                ys = [y.tolist()
                      for y in self.model.translate(sources, self.max_length)]
                hypotheses.extend(ys)

        bleu = bleu_score.corpus_bleu(
            references, hypotheses,
            smoothing_function=bleu_score.SmoothingFunction().method1)
        chainer.report({self.key[0]: bleu})

        hypotheses_r = [' '.join(map(str, y)) for y in hypotheses]
                
        scores = self.rouge.get_scores(hypotheses_r, references_r, avg=True)
        rouge_l = scores["rouge-l"]
        chainer.report({self.key[1]: rouge_l["p"]})
        chainer.report({self.key[2]: rouge_l["r"]})
        chainer.report({self.key[3]: rouge_l["f"]})

        rouge_1 = scores["rouge-1"]
        chainer.report({self.key[4]: rouge_1["p"]})
        chainer.report({self.key[5]: rouge_1["r"]})
        chainer.report({self.key[6]: rouge_1["f"]})
        
        
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
         'bleu', 'p', 'r', 'f', 'p1', 'r1', 'f1', 'elapsed_time']),
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

        @chainer.training.make_extension()
        def translate(trainer):
            source, target = test_data[numpy.random.choice(len(test_data))]
            result = model.translate([model.xp.array(source)])[0]

            #source_sentence = ' '.join([source_words[x] for x in source])
            target_sentence = ' '.join([target_words[y] for y in target])
            result_sentence = ' '.join([target_words[y] for y in result])
            #print('#  source : ' + source_sentence)
            print('#  result : ' + result_sentence)
            print('#  expect : ' + target_sentence)

        trainer.extend(
            translate, trigger=(args.validation_interval, 'iteration'))
        
        # @chainer.training.make_extension()
        # def generate(trainer):
        #     results = model.generate(5)
        #     for i, result in enumerate(results):
        #         print('#  result {}: {}'.format(i+1, ' '.join([source_words[x] for x in result])))
                
        # trainer.extend(
        #     generate, trigger=(args.validation_interval, 'iteration'))
        
        # trainer.extend(
        #     CalculateBleu(
        #         model, test_data, 'bleu', device=args.gpu),
        #     trigger=(args.validation_interval, 'iteration'))
        trainer.extend(
            CalculateBleuRouge(
                model, test_data, ['bleu', 'p', 'r', 'f', 'p1', 'r1', 'f1'], device=args.gpu),
            trigger=(args.validation_interval, 'iteration'))
        

    @chainer.training.make_extension()
    def fit_C(trainer):
        if model.C < 0.5 and updater.epoch > 5:
        #if model.C < 0.5:
            model.C += 0.001
        print('epoch: {}, C: {},'.format(updater.epoch, model.C))

    trainer.extend(fit_C, trigger=(1000, 'iteration'))
    #trainer.extend(fit_C, trigger=(1, 'epoch'))

        
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
