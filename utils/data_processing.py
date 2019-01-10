"""
 Data processing functions
 adapted from https://github.com/nyu-mll/multiNLI
"""


import numpy as np
import re
import random
import json
import collections
import nltk
import pickle
from utils.settings import CONFIG


PUNCTUATIONS = {'<', ')', '-', '*', '`', '$', '!', '/', '(', '>', '+', ':', '=', '|', '@', '#', '\\', '?', '^', '~', '%', ';', '.', '_', '}', '[', ',', "'", '&', '"', '{', ']'}


LABEL_MAP = {
    "POSITIVE": 0,
    "NEGATIVE": 1,
}

PADDING = "<PAD>"
UNKNOWN = "<UNK>"

def tokenize(text,max_length=int(CONFIG["MODEL"]["seq_length"])):
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if not(token in PUNCTUATIONS)]
    return tokens[:max_length]

def build_dictionary(training_datasets):
    """
    Extract vocabulary and build dictionary.
    """
    word_counter = collections.Counter()
    for i, dataset in enumerate(training_datasets):
        for example in dataset:
            word_counter.update(tokenize(example['premise']))
            word_counter.update(tokenize(example['hypothesis']))

    vocabulary = sorted(word_counter.keys(),key=lambda x:word_counter[x],reverse=True)
    vocabulary = [PADDING, UNKNOWN] + vocabulary

    word_indices = dict(zip(vocabulary, range(len(vocabulary))))

    return word_indices

def sentences_to_padded_index_sequences(word_indices, datasets):
    """
    Annotate datasets with feature vectors. Adding right-sided padding.
    """
    print("Padding and indexifying sentences")
    for i, dataset in enumerate(datasets):
        for example in dataset:
            for sentence in ['premise', 'hypothesis']:
                example[sentence + '_index_sequence'] = np.zeros((int(CONFIG["MODEL"]["seq_length"])), dtype=np.int32)
                token_sequence = tokenize(example[sentence])
                padding = int(CONFIG["MODEL"]["seq_length"]) - len(token_sequence)

                for i in range(int(CONFIG["MODEL"]["seq_length"])):
                    if i >= len(token_sequence):
                        index = word_indices[PADDING]
                    else:
                        if token_sequence[i] in word_indices:
                            index = word_indices[token_sequence[i]]
                        else:
                            index = word_indices[UNKNOWN]
                    example[sentence + '_index_sequence'][i] = index

def loadEmbedding_rand(path, word_indices):
    """
    Load GloVe embeddings. Doing a random normal initialization for OOV words.
    """
    n = len(word_indices)
    m = int(CONFIG["MODEL"]["word_embedding_dim"])
    emb = np.empty((n, m), dtype=np.float32)

    emb[:,:] = np.random.normal(size=(n,m))

    # Explicitly assign embedding of <PAD> to be zeros.
    emb[0:2, :] = np.zeros((1,m), dtype="float32")
    emb_to_load = int(CONFIG["MODEL"]["emb_to_load"])
    with open(path, 'r',encoding="utf-8") as f:
        for i, line in enumerate(f):
            if emb_to_load > 0:
                if i >= emb_to_load:
                    break

            line_ = line.strip().split(" ")
            if line_[0] in word_indices:
                try:
                    emb[word_indices[line_[0]], :] = np.asarray(line_[1:])
                except ValueError as e:
                    import pdb
                    pdb.set_trace()

    return emb

