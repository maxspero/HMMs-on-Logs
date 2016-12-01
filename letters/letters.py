#!/usr/bin/env python
from __future__ import division

def warn(*args, **kwargs):
        pass
import warnings
warnings.warn = warn

import numpy as np
from hmmlearn import hmm

def main():
    with open('data_letters.csv') as f:
        letters = f.readlines()
    ltr_data = [letter.split(",") for letter in letters]

    # isolate outliers
    anomalies = np.asarray([ltr for ltr in ltr_data if ltr[-1] == '"o"\n'])
    normals = np.asarray([ltr for ltr in ltr_data if ltr[-1] == '"n"\n'])

    # delete the label
    train_anomaly = np.delete(anomalies, -1, 1)
    train_normals = np.delete(normals, -1, 1)

    # retype as floats
    train_anomaly = train_anomaly.astype(np.float_, copy=False)
    train_normals = train_normals.astype(np.float_, copy=False)

    # TRAINING 2 HMMs: one for anomaly, one for normals.
    anomaly_model = hmm.GaussianHMM(n_components=2, n_iter=50)
    normal_model = hmm.GaussianHMM(n_components=2, n_iter=50)
    anomaly_model.fit(train_anomaly)
    normal_model.fit(train_normals)

    print anomaly_model.monitor_
    print normal_model.monitor_

    # TESTING:
    # Comparing model likelihoods for each anomaly and normal record.
    # Recording successful anomalies and normals
    corr_anomalies = 0
    corr_normals = 0
    for anomaly in train_anomaly:
        if anomaly_model.score(anomaly) < normal_model.score(anomaly):
            corr_anomalies += 1

    for normal in train_normals:
        if anomaly_model.score(normal) > normal_model.score(normal):
            corr_normals += 1

    # Print out anomaly, normal, and overall correctness.
    print "Anomaly Correctness rate for {0} anomalies is {1}".format(
            len(train_anomaly), 
            corr_anomalies / len(train_anomaly))

    print "Normal Correctness rate for {0} normals is {1}".format(
            len(train_normals), 
            corr_normals / len(train_normals))
    
    print "Overall Correctness rate for {0} points is {1}".format(
            len(ltr_data), 
            (corr_anomalies + corr_normals) / len(ltr_data))


if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)
    main()
