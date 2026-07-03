#!/usr/bin/env python
"""Reduced-scale but real word2vec training for CPU (documented in README).
Identical to run.py except the SGD iteration budget is 20000 instead of 40000,
which already shows clear loss convergence and produces the word-vector figure.
"""
import random, time, sys
import numpy as np
from utils.treebank import StanfordSentiment
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
from word2vec import *
from sgd import *

random.seed(314)
dataset = StanfordSentiment()
tokens = dataset.tokens()
nWords = len(tokens)
dimVectors = 10
C = 5
ITERS = 20000

random.seed(31415); np.random.seed(9265)
startTime = time.time()
wordVectors = np.concatenate(
    ((np.random.rand(nWords, dimVectors) - 0.5) / dimVectors,
     np.zeros((nWords, dimVectors))), axis=0)
wordVectors = sgd(
    lambda vec: word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, C,
                                     negSamplingLossAndGradient),
    wordVectors, 0.3, ITERS, None, True, PRINT_EVERY=10)
print("training took %d seconds" % (time.time() - startTime))

wordVectors = np.concatenate((wordVectors[:nWords, :], wordVectors[nWords:, :]), axis=0)
visualizeWords = [
    "great", "cool", "brilliant", "wonderful", "well", "amazing",
    "worth", "sweet", "enjoyable", "boring", "bad", "dumb",
    "annoying", "female", "male", "queen", "king", "man", "woman",
    "rain", "snow", "hail", "coffee", "tea"]
visualizeIdx = [tokens[w] for w in visualizeWords]
visualizeVecs = wordVectors[visualizeIdx, :]
temp = (visualizeVecs - np.mean(visualizeVecs, axis=0))
covariance = 1.0 / len(visualizeIdx) * temp.T.dot(temp)
U, S, V = np.linalg.svd(covariance)
coord = temp.dot(U[:, 0:2])
for i in range(len(visualizeWords)):
    plt.text(coord[i, 0], coord[i, 1], visualizeWords[i],
             bbox=dict(facecolor='green', alpha=0.1))
plt.xlim((np.min(coord[:, 0]), np.max(coord[:, 0])))
plt.ylim((np.min(coord[:, 1]), np.max(coord[:, 1])))
plt.title("CS224n A2: word2vec (skip-gram, neg-sampling) on SST — 20k iters")
plt.savefig('word_vectors.png')
print("saved word_vectors.png")
