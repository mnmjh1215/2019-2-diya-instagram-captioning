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
        

def generate_vocab(json_file):
    """
    returns a dictionary, contatining (word, idx) pairs for every word used in caption file
    """
    word2id = {'<PAD>': 0, '<UNK>': 1, '<start>': 2, '<end>': 3}
    idx = len(word2id)
    with open(json_file) as fr:
        item_list = json.load(fr)
        # TODO: generate vocab with given json_file
    return word2id


def load_pretrained_embedding(pretrained_path, word2id, embed_size=300):
    vocab_size = len(word2id)
    embedding = torch.randn((vocab_size, embed_size))
    with open(pretrained_path) as fr:
        # TODO: 한국어 pretrained embedding을 잘 불러오기!
        pass

    return embedding


def beam_search(decoder, encoder_output, beam_size=3):
    # TODO
    pass


def load_model(encoder, decoder, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])



