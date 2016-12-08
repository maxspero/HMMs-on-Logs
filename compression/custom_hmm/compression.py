#!/usr/bin/env python

import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder
import math
from sklearn.externals import joblib
from scipy import stats
import random
from collections import defaultdict
import itertools
import time

from utils import *

n_training_samples = 100  # None == all data
n_hidden_states = 2
n_iterations = 10
subsequence_len = 10
data_file = '../../data'

log_sequence = 17

encoder = LabelEncoder()

def format_seq(seq):
    return np.asarray(encoder.transform(seq)).reshape(-1, 1)

def extract_data(log_data, subsequence_length):
    subsequences = []
    lengths = []
    log_data = log_data[1:]
    fq = defaultdict(int)
    for log in log_data:
        if len(log) > 17:
            try:
                x = [int(i) for i in log[17].strip().split(",")]
                for s1, s2 in itertools.izip(x, x[1:]):
                    fq[(s1, s2)] += 1
                num_samples = len(x)
                if num_samples < subsequence_length:
                    continue
                for i in xrange(num_samples - subsequence_length + 1):
                    for sample in x[i:i+subsequence_length]:
                        subsequences.append([sample])
                    lengths.append(subsequence_length)
            except:
                pass 
    encoder.fit(subsequences)
    random.shuffle(subsequences)
    formatted_sequences = format_seq(subsequences)
    return formatted_sequences, lengths, fq


def train_model():
    print "Reading data..."
    with open(data_file) as f:
        logs = f.readlines()
    log_data = [log.split("\t") for log in logs]

    print "Subsequence length {}".format(subsequence_len)
    print "Extracting data..."
    training_seq, training_lengths, fq = \
        extract_data(log_data[:n_training_samples], subsequence_len)

    print "Fitting model..."
    model = hmm.MultinomialHMM(n_components=n_hidden_states, n_iter=n_iterations)
    model.fit(training_seq, training_lengths)
    joblib.dump((model, training_seq, training_lengths, fq), 'models/model2.pkl')


def score_model(modified=True):
    model, training_seq, training_lengths, fq = joblib.load('models/model2.pkl')
    scores = []
    start = time.time()
    for i in xrange(len(training_lengths)):
        seq = training_seq[i:i+subsequence_len,0]
        seq = seq.reshape(-1, 1)
        if modified:
            score = model.score_modified(seq, [subsequence_len])
        else:
            score = model.score(seq, [subsequence_len])
        scores.append(score)
    end = time.time()
    print "Time for scoring: {}".format(end-start)
    print scores[:10]


def run_all():
    print "Reading data..."
    with open(data_file) as f:
        logs = f.readlines()
    log_data = [log.split("\t") for log in logs]

    print "Subsequence length {}".format(subsequence_len)
    print "Extracting data..."
    training_seq, training_lengths, fq = \
        extract_data(log_data[:n_training_samples], subsequence_len)

    print "Fitting model..."
    model = hmm.MultinomialHMM(n_components=n_hidden_states, n_iter=n_iterations)
    model.fit(training_seq, training_lengths)

    scores = []
    start = time.time()
    for i in xrange(len(training_lengths)):
        seq = training_seq[i:i+subsequence_len,0]
        seq = seq.reshape(-1, 1)
        score = model.score_modified(seq, [subsequence_len])
        scores.append(score)
    end = time.time()
    print "Time for scoring: {}".format(end-start)
