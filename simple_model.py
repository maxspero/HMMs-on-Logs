#!/usr/bin/env python

"""
Simple model that doesn't really do anything :p
"""

import numpy as np
from hmmlearn import hmm

user_id = 0
time = 3
request_type = 4
object_id = 5
protocol = 6
status_code = 7
size = 8

def main():
    with open('test_logs') as f:
        logs = f.readlines()
    log_data = [log.replace('-', '-1').split() for log in logs]

    # client id, status, object id
    features = {200: [], 304: []}
    for data in log_data:
        features[data[status_code]].append([data[user_id], data[request_type], data[object_id], data[protocol], data[size]])

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
