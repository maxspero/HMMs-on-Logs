#!/usr/bin/env python

"""
Simple model that reads extractedLog.tsv
"""

import numpy as np
from hmmlearn import hmm
from hmmlearn import subsequencehmm
from sklearn.preprocessing import LabelEncoder
import math
from sklearn.externals import joblib

n_training_samples = 100 # None == all data
n_hidden_states = 2
n_iterations = 10
sequence_length = 15


log_sequence = 17

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

    print model.score(training_seq), model.score_modified(training_seq)
    
    print model.score(training_seq[:sequence_length])
    print model.score(training_seq[1:sequence_length+1])
    # print model.scoreSubsequences(training_seq[:sequence_length+1], sequence_length)
    print model.scoreSubsequencesNaive(training_seq[:sequence_length+1], sequence_length)
    print model.scoreSubsequencesSlidingWindow(training_seq[:sequence_length+1], sequence_length)



if __name__ == "__main__":
  main()
