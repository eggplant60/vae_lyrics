#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import sys
# sys.path.append('../')
# from seq2seq_vae import *

import tweepy

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

    # Authentication
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    # Load Model and generate lyrics
    body = generate_lyrics('/home/naoya/work/text/vae_lyrics/result_0208_C041', 1)[0]
    
    print(body)
    #api.update_status(body)
