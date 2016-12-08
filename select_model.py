#!/usr/bin/env python

"""
Simple model that reads extractedLog.tsv
"""

import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder
import math
from sklearn.externals import joblib
from scipy import stats
import random
from fractions import Fraction

from utils import *

n_training_samples = 200  # None == all data
n_test_samples = 100
#n_hidden_states = 2
n_iterations = 10
subsequence_len = 20
data_file = 'data'
num_hidden_states = [2, 3, 4, 10]

log_sequence = 17

encoder = LabelEncoder()

def format_seq(seq):
    return np.asarray(encoder.transform(seq)).reshape(-1, 1)

def extract_data(log_data, subsequence_length):
    subsequences = []
    lengths = []
    log_data = log_data[1:]
    for log in log_data:
        if len(log) > 17:
            try:
                x = [int(i) for i in log[17].strip().split(",")]
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
    return formatted_sequences, lengths


def bic_score(model, training_seq, training_lengths):
    logproba = model.score(training_seq, training_lengths)
    num_params = model.startprob_.shape[0] - 1 + \
            model.transmat_.shape[0] * (model.transmat_.shape[1] - 1) + \
            model.emissionprob_.shape[0] * (model.emissionprob_.shape[1] - 1)
    bic = -2 * logproba + num_params * math.log(len(training_lengths))
    return bic
    


def main():
    print "Reading data..."
    with open(data_file) as f:
        logs = f.readlines()
    log_data = [log.split("\t") for log in logs]

    for n_states in num_hidden_states:
        print "\nSubsequence length {}".format(subsequence_len)
        print "Extracting data..."
        training_seq, training_lengths = \
            extract_data(log_data[:n_training_samples], subsequence_len)
        test_seq, test_lengths = extract_data(
                log_data[n_training_samples:n_training_samples+n_test_samples],
                subsequence_len)

        print "Fitting model..."
        model = hmm.MultinomialHMM(n_components=n_states, n_iter=n_iterations)
        model.fit(training_seq, training_lengths)
        print "Model fit! Converged =", model.monitor_.converged

        bic = bic_score(model, test_seq, test_lengths)
        print "BIC: {}".format(bic)

        scores = []
        for i in xrange(len(test_lengths)):
            seq = test_seq[i:i+subsequence_len,0]
            seq = seq.reshape(-1, 1)
            score = model.score(seq, [subsequence_len])
            score = abs(score) ** Fraction(1, subsequence_len)
            scores.append(score)

        scores = scores[:100]
        skew = stats.skew(scores)
        kurtosis = stats.kurtosis(scores)
        k2, p = stats.normaltest(scores)
        print "skew: {}\nkurtosis: {}\nk2: {}, p: {}".format(skew, kurtosis, k2, p)



if __name__ == "__main__":
  main()
