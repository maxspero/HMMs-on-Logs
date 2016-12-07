#!/usr/bin/env python
from __future__ import division

def warn(*args, **kwargs):
        pass
import warnings
warnings.warn = warn

import numpy as np
from hmmlearn import hmm

def main():
    with open('shuttle_data.csv') as f:
        shuttles = f.readlines()
    shuttle_data = [shuttle.split(",") for shuttle in shuttles]

    # isolate outliers
    anomalies = np.asarray([shuttle for shuttle in shuttle_data if shuttle[-1] == '"o"\n'])
    normals = np.asarray([shuttle for shuttle in shuttle_data if shuttle[-1] == '"n"\n'])

    # delete the label
    anomalies = np.delete(anomalies, -1, 1).astype(np.float_, copy=False)
    normals = np.delete(normals, -1, 1).astype(np.float_, copy=False)

    # retype as floats
    train_anomaly = anomalies[:50]
    train_normals = normals[:100]

    # TRAINING 2 HMMs: one for anomaly, one for normals.
    anomaly_model = hmm.GMMHMM(n_components=2, n_iter=50)
    normal_model = hmm.GMMHMM(n_components=2, n_iter=50)
    anomaly_model.fit(train_anomaly)
    normal_model.fit(train_normals)

    print "Training stats for anomalies: \n{0}\n ".format(anomaly_model.monitor_)
    print "Training stats for normals: \n{0}\n ".format(normal_model.monitor_)

    # TESTING:
    # Comparing model likelihoods for each anomaly and normal record.
    # Recording successful anomalies and normals
    corr_anomalies = 0
    corr_normals = 0
    for anomaly in anomalies:
        if anomaly_model.score(anomaly) > normal_model.score(anomaly):
            corr_anomalies += 1

    for normal in normals:
        if anomaly_model.score(normal) < normal_model.score(normal):
            corr_normals += 1

    # Print out anomaly, normal, and overall correctness.
    print "Anomaly Correctness rate for {0} anomalies is {1}".format(
            len(anomalies), 
            corr_anomalies / len(anomalies))

    print "Normal Correctness rate for {0} normals is {1}".format(
            len(normals), 
            corr_normals / len(normals))
    
    print "Overall Correctness rate for {0} points is {1}".format(
            len(shuttle_data), 
            (corr_anomalies + corr_normals) / len(shuttle_data))

if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)
    main()
