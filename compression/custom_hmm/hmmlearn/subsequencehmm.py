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

    def scoreSubsequencesNaiveNoCython(self, X, subsequenceLength):
        check_is_fitted(self, "startprob_")
        self._check()

        X = check_array(X)
        framelogprob = self._compute_log_likelihood(X)
        outputs = self._do_forward_pass_multiframe_naive_no_cython(framelogprob, subsequenceLength)
        return [output[1] for output in outputs]

    def scoreSubsequencesNaiveModified(self, X, subsequenceLength):
        check_is_fitted(self, "startprob_")
        self._check()

        X = check_array(X)
        frameprob = self.emissionprob_[:,np.concatenate(X)].T
        outputs = self._do_forward_pass_multiframe_naive_modified(frameprob, subsequenceLength)
        return [output[1] for output in outputs]

    def scoreSubsequencesSlidingWindow(self, X, subsequenceLength):
        check_is_fitted(self, "startprob_")
        self._check()

        X = check_array(X)
        frameprob = self.emissionprob_[:,np.concatenate(X)].T
        outputs = self._do_forward_pass_sliding_window(frameprob, subsequenceLength)
        return outputs

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

    def _do_forward_pass_multiframe_naive_no_cython(self, framelogprob, framelength):
        n_samples, n_components = framelogprob.shape
        outputs = [] # array of logprob values, for logprob at i, value is prob of samples[i:i+framelength]
        for i in xrange(n_samples-framelength+1):
          fwdlattice = np.zeros((framelength, n_components))
          self._forward(framelength, n_components,
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
        
    def _do_forward_pass_sliding_window(self, frameprob, framelength):
        n_samples, n_components = frameprob.shape
        outputs = self._forward_modified_sliding_window(n_samples, n_components,
                         self.startprob_,
                         self.transmat_,
                         frameprob,
                         framelength)
        return outputs

    # doesn't work like that
    def _do_forward_pass_multiframe_matrix(self, framelogprob, framelength):
        n_samples, n_components = framelogprob.shape
        fwdlattice = np.zeros((n_samples, n_components))
        _hmmc._forward(n_samples, n_components,
                       log_mask_zero(self.startprob_),
                       log_mask_zero(self.transmat_),
                       framelogprob, fwdlattice)
        return [fwdlattice[i+framelength] - fwdlattice[i] for i in xrange(0, len(fwdlattice) - framelength)], fwdlattice

    def _forward_modified_sliding_window(self, n_samples, n_components,
                 startprob,
                 transmat,
                 frameprob,
                 framelength):
        sumlogcs = np.zeros(n_samples)
        alphas = []
        startingprobs = True
        endingprobs = False
        start = 0

        A = np.transpose(transmat)

        for i in range(0, n_samples):
            # shared for all samples in window
            B = np.diag([b for b in frameprob[i,:]])
            C = B.dot(A)
            s = np.sum(C)
            C = C/s
            logs = np.log(s)

            # initialize ith sequence
            alphas.append(B.dot(startprob))
            c = np.sum(alphas[i])
            alphas[i] = alphas[i]/c
            sumlogcs[i] += np.log(c)
            
            # continue start:i sequences
            for j in range(start,i):
              alphas[j] = C.dot(alphas[j])
              c = np.sum(alphas[j])
              alphas[j] = alphas[j]/c
              sumlogcs[j] += logs + np.log(c)

            # increment i after first framelength iterations
            if startingprobs:
              if i - start == framelength - 1:
                start += 1
                startingprobs = False
            else:
              start += 1
              
        # throw away last framelength-1 probabilities (not full sequences)
        return sumlogcs.tolist()[:n_samples-framelength+1]

    def _forward_modified_inverse(self, n_samples, n_components,
                 startprob,
                 transmat,
                 frameprob,
                 framelength):
        sumlogcs = np.zeros(n_samples)
        alphas = []
        startingprobs = True
        endingprobs = False
        start = 0

        A = np.transpose(transmat)

        for i in range(0, n_samples):
            # shared for all samples in window
            B = np.diag([b for b in frameprob[i,:]])
            C = B.dot(A)
            s = np.sum(C)
            C = C/s
            logs = np.log(s)

            # initialize ith sequence
            alphas.append(B.dot(startprob))
            c = np.sum(alphas[i])
            alphas[i] = alphas[i]/c
            sumlogcs[i] += np.log(c)
            
            # continue start:i sequences
            for j in range(start,i):
              alphas[j] = C.dot(alphas[j])
              c = np.sum(alphas[j])
              alphas[j] = alphas[j]/c
              sumlogcs[j] += logs + np.log(c)

            # increment i after first framelength iterations
            if startingprobs:
              if i - start == framelength - 1:
                start += 1
                startingprobs = False
            else:
              start += 1
              
        # throw away last framelength-1 probabilities (not full sequences)
        return sumlogcs.tolist()[:n_samples-framelength+1]


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
