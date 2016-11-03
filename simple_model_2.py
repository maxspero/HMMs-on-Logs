#!/usr/bin/env python

"""
Simple model that reads extractedLog.tsv
"""

import numpy as np
from hmmlearn import hmm

log_sequence = 17

def main():
    with open('../extractedLog.tsv') as f:
        logs = f.readlines()
    log_data = [log.split("\t") for log in logs]

    sequences = []
    lengths = []
    for log in log_data:
        if len(log) > 17:
            try:
                x = [int(i) for i in log[17].strip().split(",")]
                sequences.append(x)
                lengths.append(len(x))
            except:
                pass 

    concatSeq = np.concatenate(sequences[:-100])
    model = hmm.GaussianHMM(n_components=1, covariance_type="full", n_iter=100)
    model.fit(concatSeq, lengths[:-100])
    print model.monitor_
    print model.monitor_.converged

    X, Z = model.sample(10)
    print X, Z

    print model.predict(sequences[:-100])
    print model.score(sequences[:-100])

if __name__ == "__main__":
    main()
