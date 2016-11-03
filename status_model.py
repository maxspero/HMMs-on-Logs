#!/usr/bin/env python

"""
Simple model that doesn't really do anything :p
"""

import numpy as np
from hmmlearn import hmm

def main():
    with open('test_logs') as f:
        logs = f.readlines()
    log_data = [log.split() for log in logs]

    # status
    samples = [[1] if data[-2] == '200' else [0] for data in log_data]
    samples = np.asarray(samples).astype(int)

    observations = ["Not failure", "Failure"]
    states = ["200", "Not 200"]

    model = hmm.MultinomialHMM(n_components=2, n_iter=100, init_params='')
    model.startprob_ = np.array([0.5, 0.5])
    model.transmat_ = np.array([
        [0.8, 0.2],
        [0.2, 0.8]
    ])
    model.emissionprob_ = np.array([
        [1, 0],
        [0.5, 0.5]
    ])
    
    train_samples = samples[:990]
    model.fit(train_samples)
    print model.monitor_
    print model.monitor_.converged

    test_samples = samples[990:]
    logprob, result = model.decode(test_samples, algorithm="viterbi")
    print "Observations:", ", ".join(map(lambda x: observations[x], test_samples))
    print "States:", ", ".join(map(lambda x: states[x], result))


if __name__ == "__main__":
    main()
