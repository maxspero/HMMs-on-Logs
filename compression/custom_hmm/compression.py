#!/usr/bin/env python

import numpy as np
from hmmlearn import hmm
from hmmlearn import utils
from sklearn.preprocessing import LabelEncoder
import math
from sklearn.externals import joblib
from scipy import stats
import random
from collections import defaultdict
import itertools
import time
import operator

from utils import *

#n_training_samples = 1000  # None == all data
n_hidden_states = 2
n_iterations = 10
#subsequence_len = 10
data_file = '../../data'
#model_name = 'model2'

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
    formatted_sequences = format_seq(subsequences)
    return formatted_sequences, lengths, fq


def train_model(num_lines, subsequence_len):
    print "Reading data..."
    with open(data_file) as f:
        logs = f.readlines()
    log_data = [log.split("\t") for log in logs]

    print "Subsequence length {}".format(subsequence_len)
    print "Extracting data..."
    training_seq, training_lengths, fq = \
        extract_data(log_data[:num_lines], subsequence_len)

    print "Fitting model..."
    model = hmm.MultinomialHMM(n_components=n_hidden_states, n_iter=n_iterations)
    model.fit(training_seq, training_lengths)
    joblib.dump((model, training_seq, training_lengths, fq), 'models/' + str(num_lines) + '-' + str(subsequence_len) + '.pkl')


def encode(seq, model, fq_c):
    seq = np.concatenate(seq)
    A = np.transpose(model.transmat_)
    B = np.diag([b for b in model.emissionprob_[:, seq[0]].T])
    alpha = B.dot(model.startprob_)

    encoded_seq = []
    skip = False
    sumlogs = 0
    for a, b in itertools.izip(seq[1:], seq[2:]):
        if skip:
            skip = False
            continue
        if (a, b) in fq_c:
            C, s = fq_c[(a, b)]
            encoded_seq.append(C)
            sumlogs += s
            skip = True
        else:
            B = np.diag([b for b in model.emissionprob_[:, a].T])
            C = B.dot(A)
            s = np.sum(C)
            encoded_seq.append(C/s)
            sumlogs += np.log(s)
    if not skip:  # add last observation in seq
        B = np.diag([b for b in model.emissionprob_[:, seq[-1]].T])
        C = B.dot(A)
        s = np.sum(C)
        encoded_seq.append(C/s)
        sumlogs += np.log(s)

    return alpha, encoded_seq, sumlogs


def score_model(model_name, compressed=True, modified=True, topk=0):
    model, training_seq, training_lengths, fq = joblib.load('models/' + model_name + '.pkl')
    fq = sorted(fq.items(), key=operator.itemgetter(1), reverse=True)
    fq_c = defaultdict(float)
    A = np.transpose(model.transmat_)
    #start = time.clock()
    scores = []
    if compressed:
        for obs, _ in fq[:topk]:
            try:
                c = [0] * 3
                frameprob = model.emissionprob_[:, obs].T
                B = np.diag([b for b in frameprob[0,:]])
                C1 = B.dot(A)
                c[0] = np.sum(C1)
                #C1 = C1/c[0]
                B = np.diag([b for b in frameprob[1,:]])
                C2 = B.dot(A)
                c[1] = np.sum(C2)
                #C2 = C2/c[1]
                C_comb = C2.dot(C1)
                c[2] = np.sum(C_comb)
                C_comb = C_comb/c[2]
                #fq_c[obs] = (C_comb, np.log(np.prod(c)))
                fq_c[obs] = (C_comb, np.log(c[2]))
            except:
                pass
        start = time.clock()
        for i, j in utils.iter_from_X_lengths(training_seq, training_lengths):
            alpha, sequence, sumlogs = encode(training_seq[i:j], model, fq_c)
            score = model.score_compressed(alpha, sequence)
            scores.append(score + sumlogs)
    else:
        subsequence_len = int(model_name.split('-')[1])
        start = time.clock()
        for i, j in utils.iter_from_X_lengths(training_seq, training_lengths):
            seq = training_seq[i:j,0]
            seq = seq.reshape(-1, 1)
            if modified:
                score = model.score_modified(seq, [subsequence_len], fq_c)
            else:
                score = model.score(seq, [subsequence_len])
            scores.append(score)
    end = time.clock()
    print "Time for scoring {}: {}".format(model_name, end-start)
    #print scores[:10]
    return end-start
