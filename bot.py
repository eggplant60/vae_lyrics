#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import sys
# sys.path.append('../')
# from seq2seq_vae import *

import tweepy
import argparse

### Import the bot's information
### consumer_key, consumer_secret, access_token, access_token_secret
from password_bot import *
from extract_vector_vae import load_model_vocab



def generate_lyrics(result_dir, batchsize):

    source_ids, _, _, _, test_source, _, model = load_model_vocab(result_dir, gpu=-1)

    source_words = {i: w for w, i in source_ids.items()}

    results = model.generate(batchsize)

    lyrics = []
    for i, result in enumerate(results):
        #decode_string = ''.join([source_words[x] for x in result]).replace('/','\n')
        decode_string = words2sentence([source_words[x] for x in result])
        lyrics.append(decode_string)
    
    return lyrics


def is_alphabet(word):
    return all([65 <= ord(char) <= 122  for char in word])


def words2sentence(words):
    len_words = len(words)
    sentence = words[0]
    for i in range(len_words-1):
        if words[i+1] == '/':
            sentence += '\n'
        elif is_alphabet(words[i]) \
             and is_alphabet(words[i+1]) \
             and words[i] != '/':
            sentence += (' ' + words[i+1])
        else:
            sentence += words[i+1]
    return sentence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='search lyrics similar to the query')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='generate sentence\(s\) without tweet')
    parser.add_argument('--num', '-n', type=int, default=1,
                        help='number of sentences generated')
    args = parser.parse_args()

    # Load Model and generate lyrics
    bodies = generate_lyrics('/home/naoya/work/text/vae_lyrics/result_0209_C024', args.num)
    for i in range(args.num):
        print(bodies[i])
        print()
        
    if not args.quiet:
        # Authentication
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        api = tweepy.API(auth)
        api.update_status(bodies[0])
