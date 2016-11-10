#!/usr/bin/env python

"""
Simple model that reads extractedLog.tsv
"""

import numpy as np
from hmmlearn import hmm

log_sequence = 17


def extract_data(log_data):
    sequences = []
    lengths = []
    for log in log_data:
        if len(log) > 17:
            try:
                x = [int(i) for i in log[17].strip().split(",")]
                for i in x:
                    sequences.append([i])
                lengths.append(len(x))
            except:
                pass 
    return np.asarray(sequences), lengths


def main():
    with open('../extractedLog.tsv') as f:
        logs = f.readlines()
    log_data = [log.split("\t") for log in logs]

    training_seq, training_lengths = extract_data(log_data[:100])

    model = hmm.MultinomialHMM(n_components=2, n_iter=10)
    model.fit(training_seq, training_lengths)
    print model.monitor_
    print model.monitor_.converged

    #X, Z = model.sample(10)
    #print X, Z

    test_seq, test_lengths = extract_data(log_data[:100])

    posteriors = model.predict_proba(test_seq, test_lengths)
    print np.around(posteriors, 2)

if __name__ == "__main__":
    main()
