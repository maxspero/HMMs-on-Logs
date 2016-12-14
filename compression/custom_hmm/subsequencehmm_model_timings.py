#!/usr/bin/env python

"""
Simple model that reads extractedLog.tsv
"""

import numpy as np
from hmmlearn import hmm
from hmmlearn import subsequencehmm
from sklearn.preprocessing import LabelEncoder
import math
import time
from sklearn.externals import joblib

n_training_samples = 100 # None == all data
n_hidden_states = 2
n_iterations = 10
sequence_length = 50


encoder = LabelEncoder()

def format_seq(seq):
  return np.asarray(encoder.transform(seq)).reshape(-1, 1)

def extract_data(log_data):
    sequences = []
    lengths = []
    for log in log_data:
        if len(log) > 17:
            try:
                x = [int(i) for i in log[17].strip().split(",")]
                if len(x) == 0: 
                  continue
                for i in x:
                    sequences.append([i])
                lengths.append(len(x))
            except:
                pass 
    encoder.fit(sequences)
    formatted_sequences = format_seq(sequences)
    return formatted_sequences, lengths


def main():
    print "Reading data..."
    with open('../../../extractedLog.tsv') as f:
        logs = f.readlines()
    log_data = [log.split("\t") for log in logs]

    print "Extracting data..."
    training_seq, training_lengths = extract_data(log_data[:n_training_samples])

    print "Fitting model..."
    model = subsequencehmm.SubsequenceHMM(n_components=n_hidden_states, n_iter=n_iterations)
    model.fit(training_seq, training_lengths)
    print "Model fit! Converged =", model.monitor_.converged

    scores = []
    sequences = []
    for i in xrange(len(training_seq[:,0])):
        seq = training_seq[i:i+sequence_length,0]
        sequences.append(seq)
        seq = seq.reshape(-1, 1)
        score = model.score(seq)
        scores.append(score)

    results = [[score, seq] for score, seq in zip(scores, sequences)]
    results = sorted(results, key=lambda x: x[0])

    subseq_scores = model.scoreSubsequencesNaive(training_seq, sequence_length)

    # print scores, subseq_scores
    print len(scores), len(subseq_scores)
    print scores[-1], subseq_scores[-1]

    start = time.clock()
    out_sliding_window = model.scoreSubsequencesSlidingWindow(training_seq, sequence_length)
    end = time.clock()
    print "sliding window: %f seconds. (max %f, min %f)" % (end - start, max(out_sliding_window), min(out_sliding_window))
    start = time.clock()
    out_naive_modified = model.scoreSubsequencesNaiveModified(training_seq, sequence_length)
    end = time.clock()
    print "naive modified: %f seconds. (max %f, min %f)" % (end - start, max(out_naive_modified), min(out_naive_modified))
    start = time.clock()
    out_naive_no_c = model.scoreSubsequencesNaiveNoCython(training_seq, sequence_length)
    end = time.clock()
    print "naive no cython: %f seconds. (max %f, min %f)" % (end - start, max(out_naive_no_c), min(out_naive_no_c))
    start = time.clock()
    out_naive = model.scoreSubsequencesNaive(training_seq, sequence_length)
    end = time.clock()
    print "naive cython: %f seconds. (max %f, min %f)" % (end - start, max(out_naive), min(out_naive))



if __name__ == "__main__":
  main()
