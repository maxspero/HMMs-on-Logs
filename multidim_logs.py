#!/usr/bin/env python

import numpy as np
from hmmlearn import hmm

LOG_SEQ_INDEX = 17
LIB_VERSION_INDEX = 7

lib_version_map = {}

"""
This maps a lib_version to a unique integer identifier
"""
def get_version_mapping(lib_version):
    if lib_version in lib_version_map:
        return lib_version_map[lib_version]
    else:
        next_mapping = 0
        if lib_version_map:
            next_mapping = max(lib_version_map.values()) + 1
        lib_version_map[lib_version] = next_mapping
        return next_mapping

"""
Parses collection of records by log sequence and each sequence item's version.
Returns a Nx2 array, each row is a log and the lib_version, and 
N is the number of logs seen. 
Also returns a list of lengths of sequences (offsets
into the logs array).
"""
def extract_data(records):
    sequences = []
    versions = []
    lengths = []
    for record in records:
        if len(record) > 17:
            try:
                log_sequence = [int(i) for i in record[LOG_SEQ_INDEX].strip().split(",")]
                sequences.extend(log_sequence)

                version_mapping = get_version_mapping(record[LIB_VERSION_INDEX])
                version_sequence = len(log_sequence) * [version_mapping]
                versions.extend(version_sequence)

                lengths.append(len(log_sequence))
            except:
                pass

    return np.column_stack([sequences, versions]), lengths

def main():
    with open('../extractedLog.tsv') as f:
        logs = f.readlines()
    log_data = [log.split("\t") for log in logs]
    
    # TRAINING
    training_seq, training_lengths = extract_data(log_data[:100])
    model = hmm.GaussianHMM(n_components=2, n_iter=10)
    model.fit(training_seq, training_lengths)

    print model.monitor_
    print model.monitor_.converged

    # TESTING
    test_seq, test_lengths = extract_data(log_data[:100])
    posteriors = model.predict_proba(test_seq, test_lengths)
    print np.around(posteriors, 2)


if __name__ == "__main__":
    # np.set_printoptions(threshold=np.inf)  // print all elems in array
    main()
