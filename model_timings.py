#!/usr/bin/env python

"""
Simple model that reads extractedLog.tsv
"""

import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder
import math
from sklearn.externals import joblib
import cProfile
import time

n_training_samples = -1
n_hidden_states = 3
n_iterations = 10
savefile = "model_save_and_load.hmm"
#loadfile = "model_save_and_load.hmm"
loadfile = None


log_sequence = 17


encoder = LabelEncoder()

def format_seq(seq):
  return np.asarray(encoder.transform(seq)).reshape(-1, 1)

def extract_data(log_data):
    sequences = []
    lengths = []
    for log in log_data:
        if len(log) > 17:
            try:
                x = [int(i) for i in log[17].strip().split(",")]
                if len(x) == 0: 
                  continue
                for i in x:
                    sequences.append([i])
                lengths.append(len(x))
            except:
                pass 
    encoder.fit(sequences)
    formatted_sequences = format_seq(sequences)
    return formatted_sequences, lengths


def main():
    if loadfile is None:
      with open('../extractedLog.tsv') as f:
          logs = f.readlines()
      log_data = [log.split("\t") for log in logs]

      training_seq, training_lengths = extract_data(log_data[:n_training_samples])

      model = hmm.MultinomialHMM(n_components=n_hidden_states, n_iter=n_iterations)
      model.fit(training_seq, training_lengths)
      joblib.dump(model, savefile)
      print "Model saved as", savefile 
    else:
      model = joblib.load(loadfile)
      print "Model loaded from", loadfile
    print model.monitor_
    print model.monitor_.converged

    # todo: save fitted LabelEncoder
    outs = []
    timings = []
    for x in range(195):
      start = time.time()
      seq = format_seq([x])
      outs.append((x, model.score(seq)))
      end = time.time()
      timings.append(end - start)
    print "average time to score: ", sum(timings)/len(timings)
    print "max/min time to score: ", max(timings), min(timings)
    sorted_outs = sorted(outs, key=lambda out: out[1])
    prob_sum = 0
    for out in sorted_outs:
      print out, math.exp(out[1])
      prob_sum += math.exp(out[1])
    print prob_sum
    outs = []
    timings = []
    for x in training_seq[:500]:
      start = time.time()
      seq = format_seq(x)
      outs.append((x, model.score(seq)))
      end = time.time()
      timings.append(end - start)
    print "average time to score: ", sum(timings)/len(timings)
    print "max/min time to score: ", max(timings), min(timings)
    print "average length", sum([len(x) for x in training_seq[:500]])/500
    sorted_outs = sorted(outs, key=lambda out: out[1])
    for out in sorted_outs:
      print out, math.exp(out[1])
    import random
    timings = []
    for x in range(100):
      seq = [random.randrange(80,100) for _ in range(1000)]
      start = time.time()
      seq = format_seq(seq)
      outs.append((x, model.score(seq)))
      end = time.time()
      timings.append(end - start)
    print "time for 1000 length sequence: ", sum(timings)/len(timings)
    timings = []
    for x in range(100):
      seq = [random.randrange(80,100) for _ in range(10000)]
      start = time.time()
      seq = format_seq(seq)
      outs.append((x, model.score(seq)))
      end = time.time()
      timings.append(end - start)
    print "time for 10000 length sequence: ", sum(timings)/len(timings)



if __name__ == "__main__":
  cProfile.run('main()')
