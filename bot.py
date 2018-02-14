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
from extract_vector_vae import *


def generate_lyrics(result_dir, batchsize):

    source_ids, _, _, _, test_source, _, model = load_model_vocab(result_dir)

    source_words = {i: w for w, i in source_ids.items()}

    results = model.generate(batchsize)

    lyrics = []
    for i, result in enumerate(results):
        decode_string = ''.join([source_words[x] for x in result]).replace('/','\n')
        lyrics.append(decode_string)
    
    return lyrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='search lyrics similar to the query')
    parser.add_argument('--quiet', '-q', action='store_true')
    parser.add_argument('--num', '-n', type=int, default=1)
    args = parser.parse_args()

    # Load Model and generate lyrics
    bodies = generate_lyrics('/home/naoya/work/text/vae_lyrics/result_0208_C041', args.num)
    for i in range(args.num):
        print(bodies[i])
        print()
        
    if not args.quiet:
        # Authentication
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        api = tweepy.API(auth)
        api.update_status(bodies[0])
