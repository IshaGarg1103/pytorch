import collections
import random
import re
import os
import torch
import urllib.request

import matplotlib.pyplot as plt

DATA_URL = "http://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt"
DATA_DIR = "./rnn-pytorch/text-seq/data"
class TimeMachine:
    def __init__(self, root = DATA_DIR):
        self.root = root
        os.makedirs(self.root, exist_ok = True)

    def _download(self):
        fname = os.path.join(self.root, "timemachine.txt")
        if not os.path.exists(fname):
            urllib.request.urlretrieve(DATA_URL, fname)
        with open(fname,'r',encoding="utf-8") as f:
            return f.read()
    def _preprocess(self, text):
        return re.sub('[^A-Za-z]+',' ',text).lower()
    def _tokenize(self,text):
        return list(text)
    def build(self, raw_text, vocab = None):
        tokens = self._tokenize(self._preprocess(raw_text))
        if vocab is None: vocab = Vocab(tokens)
        corpus = [vocab[token] for token in tokens]
        return corpus, vocab
    
   

        
data = TimeMachine()
raw_text = data._download()
text = data._preprocess(raw_text)
tokens = data._tokenize(text)
#print(','.join(tokens[:30]))

class Vocab:
    def __init__(self, tokens=[],min_freq = 0, reserved_tokens=[]):
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key = lambda x: x[1], reverse= True )

        #list of unique tokens
        self.idx_to_token = list(sorted(set(['<unk>'] + reserved_tokens + [
            token for token, freq in self.token_freqs if freq >= min_freq])))
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
    def __len__(self):
        return len(self.idx_to_token)
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    def to_tokens(self, indices):
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]
    @property
    def unk(self):
        return self.token_to_idx['<unk>']
vocab = Vocab(tokens)
indices = vocab[tokens[:10]]
# print('indices:', indices)
# print('tokens:', vocab.to_tokens(indices))
# corpus, vocab = data.build(raw_text)
# print(len(corpus), len(vocab))

words = text.split()
vocab = Vocab(words)
#print(vocab.token_freqs[:10])

freqs = [freq for token, freq in vocab.token_freqs]
# plt.figure(figsize=(6,3))
# plt.plot(freqs)
# plt.xlabel('token: x')
# plt.ylabel('frequency: n(x)')
# plt.xscale('log')
# plt.yscale('log')
# plt.grid(True)
# plt.show()

bigram_tokens = ['--'.join(pair) for pair in zip(words[:-1], words[1:])]
bigram_vocab = Vocab(bigram_tokens)
print(bigram_vocab.token_freqs[:10])

trigram_tokens = ['--'.join(triple) for triple in zip(words[:-2], words[1:-1], words[2:])]
trigram_vocab = Vocab(trigram_tokens)
print(trigram_vocab.token_freqs[:10])

bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
plt.figure(figsize=(6,3))
plt.plot(bigram_freqs, label='bigram')
plt.plot(trigram_freqs, label='trigram')
plt.xlabel('token: x')
plt.ylabel('frequency: n(x)')
plt.xscale('log')
plt.yscale('log')
plt.grid(True)
plt.show()