#!/usr/bin/env python

"""
Simple model that doesn't really do anything :p
"""

import numpy as np
from hmmlearn import hmm


def main():
    with open('test_logs') as f:
        logs = f.readlines()
    log_data = [log.replace('-', '-1').split() for log in logs]

    # client id, status, object id
    features = [[data[0], data[-2], data[-1]] for data in log_data]

    samples = np.asarray(features).astype(np.float)
    model = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=100)
    model.fit(samples[:800,:])
    print model.monitor_
    print model.monitor_.converged

    X, Z = model.sample(10)
    print X, Z

    print model.predict(samples[800:,:])
    print model.score(samples[800:,:])


if __name__ == "__main__":
    main()
