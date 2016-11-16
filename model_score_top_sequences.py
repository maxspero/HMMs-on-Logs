#!/usr/bin/env python

"""
Simple model that reads extractedLog.tsv
"""

import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder
import math
from sklearn.externals import joblib

n_training_samples = None # None == all data
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
    with open('../extractedLog.tsv') as f:
        logs = f.readlines()
    log_data = [log.split("\t") for log in logs]

    print "Extracting data..."
    training_seq, training_lengths = extract_data(log_data[:n_training_samples])

    print "Fitting model..."
    model = hmm.MultinomialHMM(n_components=n_hidden_states, n_iter=n_iterations)
    model.fit(training_seq, training_lengths)
    print "Model fit! Converged =", model.monitor_.converged

    scores = []
    sequences = []
    for i in xrange(len(training_lengths)):
        seq = training_seq[i:i+sequence_length,0]
        sequences.append(seq)
        seq = seq.reshape(-1, 1)
        score = model.score(seq, [sequence_length])
        scores.append(score)

    results = [[score, seq] for score, seq in zip(scores, sequences)]
    results = sorted(results, key=lambda x: x[0])

    with open('score_top_sequences_results', 'w') as f:
        f.write("Lowest probability sequences\n")
        for result in results[:50]:
            f.write("{}: {}\n".format(result[0], result[1]))
        f.write("\nHighest probability sequences\n")
        for result in results[-50:]:
            f.write("{}: {}\n".format(result[0], result[1]))


if __name__ == "__main__":
  main()
