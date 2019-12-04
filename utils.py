import torch
import torch.nn as nn
import argparse
from config import Config
import torchvision.utils as tvutils
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
import jpype
import json
from konlpy import jvm
from konlpy.tag import Okt
from collections import Counter


okt = Okt()
def tokenize_fn(text):
    tokens = okt.pos(text, norm=True, join=True)
    return tokens


class OktDetokenizer:
    def __init__(self, jvm_path=None, max_heap_size=1024):
        if not jpype.isJVMStarted():
            jvm.init_jvm(jvm_path, max_heap_size)
            
        oktPackage = jpype.JPackage('org.openkoreantext.processor') 
        self.processor = oktPackage.OpenKoreanTextProcessorJava()
        
    def detokenize(self, tokens):
        jTokens = jpype.java.util.ArrayList()
        
        for token in tokens:
            if type(token) == tuple:
                jTokens.add(token[0])
            elif type(token) == str:
                jTokens.add(token.split('/')[0])
                
        return self.processor.detokenize(jTokens)
    

def generate_vocab(words, min_freq=5):
    """
    words: list of all words appearance
    returns a dictionary, contatining (word, idx) pairs for every word used in caption file
    """
    word2id = {'<PAD>': 0, '<UNK>': 1, '<start>': 2, '<end>': 3}
    idx = len(word2id)
    counter = Counter(words)
    for word, count in counter.most_common():
        if count < min_freq:
            break
        word2id[word] = idx
        idx += 1
    
    return word2id


def load_pretrained_embedding(vocab):
    # load fasttext cc pretraiend word embedding
    if not os.path.isfile('embedding/wiki.ko.vec'):
        if not os.path.isdir('embedding'):
            os.mkdir('embedding')
        subprocess.run(['wget', '--show-progress', '-P', 'embedding', 'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.ko.vec', ])

    word2idx = dict([(word.split('/')[0], idx) for word, idx in vocab.items()])
    
    with open('embedding/wiki.ko.vec') as fr:
        embed_vocab_size, embed_dim = map(int, fr.readline().split())
        embedding_mat = np.random.randn(len(vocab), embed_dim)
        count = 0
        for line in fr:
            line = line.split()
            token = line[0]
            if token in word2idx:
                index = word2idx[token]
                vector = np.asarray(list(map(float, line[1:])))
                embedding_mat[index] = vector
                count += 1
            
    print('{0} / {1} word vectors updated'.format(count, len(vocab)))
        
    return torch.FloatTensor(embedding_mat)


def load_model(encoder, decoder, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])



