import math
from heapq import heappush, heappop, heapify
from collections import defaultdict


def bic_score(model, training_seq, training_lengths):
    logproba = model.score(training_seq, training_lengths)
    num_params = model.startprob_.shape[0] - 1 + \
            model.transmat_.shape[0] * (model.transmat_.shape[1] - 1) + \
            model.emissionprob_.shape[0] * (model.emissionprob_.shape[1] - 1)
    bic = -2 * logproba + num_params * math.log(len(training_lengths))
    return bic


def get_frequency(samples):
    fq = defaultdict(int)
    for s in samples:
        fq[s] += 1


def huffman_encode(symb2freq):
    """Huffman encode the given dict mapping symbols to weights"""
    heap = [[wt, [sym, ""]] for sym, wt in symb2freq.items()]
    heapify(heap)
    while len(heap) > 1:
        lo = heappop(heap)
        hi = heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return sorted(heappop(heap)[1:], key=lambda p: (len(p[-1]), p))
