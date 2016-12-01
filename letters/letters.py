#!/usr/bin/env python

import numpy as np
from hmmlearn import hmm

def main():
    with open('data_letters.csv') as f:
        letters = f.readlines()
    ltr_data = [letter.split(",") for letter in letters]
    
    # TRAINING
    training_data = np.asarray([ltr[:-1] for ltr in ltr_data], dtype=np.float_)
    print 'Training data has size %s' % (training_data.shape,)

    model = hmm.GaussianHMM(n_components=4, n_iter=100)   # 3 normals and anomalies.
    model.fit(training_data[:100])

    print model.monitor_
    print model.monitor_.converged

    # TESTING
    posteriors = model.predict_proba(training_data[:100])
    print np.around(posteriors, 2)

    
if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)
    main()
