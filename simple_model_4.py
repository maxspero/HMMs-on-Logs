#!/usr/bin/env python

"""
Simple model that reads extractedLog.tsv
"""

import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt

log_sequence = 17
sequence_length = 10  # make this a command-line arg


def extract_data(log_data):
    sequences = []
    lengths = []
    num_less_than_seqlen = 0
    for log in log_data:
        if len(log) > 17:
            try:
                x = [int(i) for i in log[17].strip().split(",")]
                num_samples = len(x)
                if num_samples < sequence_length:
                    num_less_than_seqlen += 1
                    continue
                for i in xrange(num_samples - sequence_length + 1):
                    for sample in x[i:i+sequence_length]:
                        sequences.append([sample])
                    lengths.append(sequence_length)
            except:
                pass
    print len(sequences)
    print num_less_than_seqlen
    return np.asarray(sequences), lengths


def main():
    with open('data') as f:
        logs = f.readlines()
    log_data = [log.split("\t") for log in logs]

    training_seq, training_lengths = extract_data(log_data[:1000])

    model = hmm.MultinomialHMM(n_components=2, n_iter=10)
    model.fit(training_seq, training_lengths)
    print model.monitor_
    print model.monitor_.converged

    #test_seq, test_lengths = extract_data(log_data[:100])

    #posteriors = model.predict_proba(test_seq, test_lengths)
    #posteriors = np.around(posteriors, 2)

    scores = []
    sequences = []
    for i in xrange(len(training_lengths)):
        seq = training_seq[i:i+sequence_length,0]
        sequences.append(seq)
        seq = seq.reshape(-1, 1)
        score = model.score(seq, [sequence_length])
        scores.append(score)

    plt.hist(scores, 50)
    plt.show(block=True)
    plt.savefig('model_4_fig.png')

    results = [[score, seq] for score, seq in zip(scores, sequences)]
    results = sorted(results, key=lambda x: x[0])

    with open('model_4_results', 'w') as f:
        f.write("Lowest probability sequences\n")
        for result in results[:50]:
            f.write("{}: {}\n".format(result[0], result[1]))
        f.write("\nHighest probability sequences\n")
        for result in results[-50:]:
            f.write("{}: {}\n".format(result[0], result[1]))


if __name__ == "__main__":
    main()
