from __future__ import print_function

import string
import sys
from collections import deque

import numpy as np
from scipy.misc import logsumexp
from sklearn.base import BaseEstimator, _pprint
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_is_fitted

from . import _hmmc
from .utils import normalize, log_normalize, iter_from_X_lengths, log_mask_zero
from .hmm import MultinomialHMM


class SubsequenceHMM(MultinomialHMM):

    # incorrect - also takes into account previous subsequences
    def scoreSubsequences(self, X, subsequenceLength):
        check_is_fitted(self, "startprob_")
        self._check()

        X = check_array(X)
        # XXX we can unroll forward pass for speed and memory efficiency.
        framelogprob = self._compute_log_likelihood(X)
        prob, fwdlattice  = self._do_forward_pass(framelogprob)
        return [logsumexp(fwdlattice[i]) for i in xrange(subsequenceLength-1, len(fwdlattice))]

    def scoreSubsequencesNaive(self, X, subsequenceLength):
        check_is_fitted(self, "startprob_")
        self._check()

        X = check_array(X)
        framelogprob = self._compute_log_likelihood(X)
        outputs = self._do_forward_pass_multiframe_naive(framelogprob, subsequenceLength)
        return [output[1] for output in outputs]

    def scoreSubsequencesNaiveModified(self, X, subsequenceLength):
        check_is_fitted(self, "startprob_")
        self._check()

        X = check_array(X)
        frameprob = self.emissionprob_[:,np.concatenate(X)].T
        outputs = self._do_forward_pass_multiframe_naive_modified(frameprob, subsequenceLength)
        return [output[1] for output in outputs]

    def _do_forward_pass(self, framelogprob):
        n_samples, n_components = framelogprob.shape
        fwdlattice = np.zeros((n_samples, n_components))
        _hmmc._forward(n_samples, n_components,
                       log_mask_zero(self.startprob_),
                       log_mask_zero(self.transmat_),
                       framelogprob, fwdlattice)
        return logsumexp(fwdlattice[-1]), fwdlattice

    def _do_forward_pass_multiframe_naive(self, framelogprob, framelength):
        n_samples, n_components = framelogprob.shape
        outputs = [] # array of logprob values, for logprob at i, value is prob of samples[i:i+framelength]
        for i in xrange(n_samples-framelength+1):
          fwdlattice = np.zeros((framelength, n_components))
          _hmmc._forward(framelength, n_components,
                         log_mask_zero(self.startprob_),
                         log_mask_zero(self.transmat_),
                         framelogprob[i:i+framelength], fwdlattice)
          outputs.append((fwdlattice, logsumexp(fwdlattice[-1])))
        return outputs

    def _do_forward_pass_multiframe_naive_modified(self, frameprob, framelength):
        n_samples, n_components = frameprob.shape
        outputs = [] # array of logprob values, for logprob at i, value is prob of samples[i:i+framelength]
        for i in xrange(n_samples-framelength+1):
          fwdlattice = np.zeros((framelength, n_components))
          prob = self._forward_modified(framelength, n_components,
                         self.startprob_,
                         self.transmat_,
                         frameprob[i:i+framelength], fwdlattice)
          outputs.append((fwdlattice, prob))
        return outputs

    # ?
    def _do_forward_pass_multiframe_matrix(self, framelogprob, framelength):
        n_samples, n_components = framelogprob.shape
        fwdlattice = np.zeros((n_samples, n_components))
        _hmmc._forward(n_samples, n_components,
                       log_mask_zero(self.startprob_),
                       log_mask_zero(self.transmat_),
                       framelogprob, fwdlattice)
        return [fwdlattice[i+framelength] - fwdlattice[i] for i in xrange(0, len(fwdlattice) - framelength)], fwdlattice

    # ?
    def _do_forward_pass_multiframe_loops(self, framelogprob, framelength):
        n_samples, n_components = framelogprob.shape
        outputs = [] # array of logprob values, for logprob at i, value is prob of samples[i:i+framelength]
        for i in xrange(n_samples-framelength+1):
          fwdlattice = np.zeros((framelength, n_components))
          _hmmc._forward(framelength, n_components,
                         log_mask_zero(self.startprob_),
                         log_mask_zero(self.transmat_),
                         framelogprob[i:i+framelength], fwdlattice)
          outputs.append((fwdlattice, logsumexp(fwdlattice[-1])))
        return outputs

    # def _compute_log_likelihood(self, X):
        """Computes per-component log probability under the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.

        Returns
        -------
        logprob : array, shape (n_samples, n_components)
            Log probability of each sample in ``X`` for each of the
            model states.

    def score(self, X, lengths=None):
        check_is_fitted(self, "startprob_")
        self._check()

        X = check_array(X)
        # XXX we can unroll forward pass for speed and memory efficiency.
        logprob = 0
        for i, j in iter_from_X_lengths(X, lengths):
            framelogprob = self._compute_log_likelihood(X[i:j])
            logprobij, _fwdlattice = self._do_forward_pass(framelogprob)
            logprob += logprobij
        return logprob
        """
