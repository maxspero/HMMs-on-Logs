#!/usr/bin/env python

"""
Simple model that reads extractedLog.tsv
"""

import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder
import math

n_training_samples = 500
n_hidden_states = 3
n_iterations = 10




log_sequence = 17

start_code = 500
end_code = 501

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
                sequences.append([start_code])
                for i in x:
                    sequences.append([i])
                sequences.append([end_code])
                lengths.append(len(x)+2)
            except:
                pass 
    encoder.fit(sequences)
    formatted_sequences = format_seq(sequences)
    return formatted_sequences, lengths


def main():
    with open('../extractedLog.tsv') as f:
        logs = f.readlines()
    log_data = [log.split("\t") for log in logs]

    training_seq, training_lengths = extract_data(log_data[:n_training_samples])

    model = hmm.MultinomialHMM(n_components=n_hidden_states, n_iter=n_iterations)
    model.fit(training_seq, training_lengths)
    print model.monitor_
    print model.monitor_.converged

    """
    test_seq, test_lengths = extract_data(log_data[:100])

    posteriors = model.predict_proba(test_seq, test_lengths)
    print np.around(posteriors, 2)
    """
    outs = []
    for x in range(195):
      seq = format_seq([x])
      outs.append((x, model.score(seq)))
    sorted_outs = sorted(outs, key=lambda out: out[1])
    prob_sum = 0
    for out in sorted_outs:
      print out, math.exp(out[1])
      prob_sum += math.exp(out[1])
    print prob_sum
    seq = format_seq([start_code])
    print "start:", model.score(seq)
    seq = format_seq([end_code])
    print "end:", model.score(seq)
    seq = format_seq([start_code, end_code])
    print "start end:", model.score(seq)
      
    """
    print "......................."
    outs = []
    for x in range(100):
      seq = format_seq([x, end_code])
      outs.append((x, model.score(seq)))
    sorted_outs = sorted(outs, key=lambda out: out[1])
    for out in sorted_outs:
      print out
    """



if __name__ == "__main__":
    main()
