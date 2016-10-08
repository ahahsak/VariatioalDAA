import numpy as np
from numpy.random import rand, dirichlet, normal, random, randn
from scipy.cluster import vq
from scipy.special import gammaln, digamma
from scipy.linalg import eig, inv, cholesky
from vbhmm.util import *

"""
    HMM with Gaussian emission probability.
"""


class GaussianHmmBase():

    def __init__(self, N, UPI0=0.5, UA0=0.5, M0=0.0,
                 BETA0=1, NU0=1, S0=0.01):

        self.n_states = N
        # log initial probability
        self._lnpi = np.log(np.tile(1.0 / N, N))
        # log transition probability
        self._lnA = np.log(dirichlet([1.0] * N, N))

    def _initialize_hmm(self, obs, scale=10.0):
        n_states = self.n_states
        self.A = dirichlet([1.0] * n_states, n_states)   # A:状態遷移行列
        self.pi = np.tile(1.0 / n_states, n_states)  # pi:初期状態確率

        T, D = obs.shape
        self.mu, temp = vq.kmeans2(obs, n_states)
        self.cv = np.tile(np.identity(D), (n_states, 1, 1))

        """
        if self._nu0 < D:
            self._nu0 += D

        self._m0 = np.mean(obs, 0)
        self._v0 = np.atleast_2d(np.cov(obs.T)) * scale

        # posterior for hidden states
        self.z = dirichlet(np.tile(1.0 / n_states, n_states), T)
        # for mean vector
        self._m, temp = vq.kmeans2(obs, n_states, minit='points')
        self._beta = np.tile(self._beta0, n_states)
        # for covarience matrix
        self._v = np.tile(np.array(self._v0), (n_states, 1, 1))
        self._nu = np.tile(float(T) / n_states, n_states)

        # aux valable
        self._c = np.array(self._v)
        """

    def _allocate_fb(self, obs):
        # fbアルゴリズムを走らせた時の一時保存用
        T = len(obs)
        lnalpha = np.zeros((T, self.n_states))  # log forward variable
        lnbeta = np.zeros((T, self.n_states))  # log backward variable
        lnxi = np.zeros((T - 1, self.n_states, self.n_states))
        return lnalpha, lnbeta, lnxi

    def _forward(self, lnf, lnalpha):
        """
        Use forward algorith to calculate forward variables and loglikelihood
        input
          lnF [ndarray, shape (n,n_states)] : loglikelihood of emissions
          lnAlpha [ndarray, shape (n,n_states)] : log forward variable
        output
          lnAlpha [ndarray, shape (n,n_states)] : log forward variable
          lnP [float] : lnP(X|theta)
        """
        T = len(lnf)
        lnalpha *= 0.0
        lnalpha[0, :] = self._lnpi + lnf[0, :]

        for t in range(1, T):
            lnalpha[t, :] = logsum(lnalpha[t - 1, :] +
                                   self._lnA.T, 1) + lnf[t, :]

        return lnalpha, logsum(lnalpha[-1, :])

    def _backward(self, lnf, lnbeta):
        """
        Use backward algorith to calculate backward variables and loglikelihood
        input
            lnF [ndarray, shape (n,n_states)] : loglikelihood of emissions
            lnBeta [ndarray, shape (n,n_states)] : log backward variable
        output
            lnBeta [ndarray, shape (n,n_states)] : log backward variable
            lnP [float] : lnP(X|theta)
        """
        T = len(lnf)
        lnbeta[T - 1, :] = 0.0

        for t in range(T - 2, -1, -1):
            lnbeta[t, :] = logsum(
                self._lnA + lnf[t + 1, :] + lnbeta[t + 1, :], 1)

        return lnbeta, logsum(lnbeta[0, :] + lnf[0, :] + self._lnpi)

    def _eval_hidden_states(self, obs):
        """
        Estep
        Then obtain variational free energy and posterior over hidden states
        """

        lnf = self._log_like_f(obs)
        lnalpha, lnbeta, lnxi = self._allocate_fb(obs)
        lnxi, lngamma, lnp = self._e_step(lnf, lnalpha, lnbeta, lnxi)
        z = np.exp(lngamma)
        return z, lnp

    def _e_step(self, lnf, lnalpha, lnbeta, lnxi):
        """
        lnF [ndarray, shape (n,n_states)] : loglikelihood of emissions
        lnAlpha [ndarray, shape (n, n_states]: log forward message
        lnBeta [ndarray, shape (n, n_states)]: log backward message
        lnPx_f: log sum of p(x_n) by forward message for scalling
        lnPx_b: log sum of p(x_n) by backward message for scalling
        """
        T = len(lnf)
        # forward-backward algorithm
        lnalpha, lnpx_f = self._forward(lnf, lnalpha)
        lnbeta, lnpx_b = self._backward(lnf, lnbeta)

        # check if forward and backward were done correctly
        dlnp = lnpx_f - lnpx_b
        if abs(dlnp) > 1.0e-6:
            print("warning forward and backward are not equivalent")

        # compute lnXi for updating transition matrix
        for i in range(self.n_states):
            for j in range(self.n_states):
                for t in range(T - 1):
                    lnxi[t, i, j] = lnalpha[t, i] + self._lnA[i, j, ] + \
                        lnf[t + 1, j] + lnbeta[t + 1, j]
        lnxi -= lnpx_f

        # compute lnGamma for postetior on hidden states
        lngamma = lnalpha + lnbeta - lnpx_f

        return lnxi, lngamma, lnpx_f

    def _m_step(self, obs, lnxi, lngamma):
        self._calculate_sufficient_statistics(obs, lnxi, lngamma)
        self._update_parameters(obs, lnxi, lngamma)

    def _calculate_sufficient_statistics(self, obs, lnxi, lngamma):
        pass

    def _update_parameters(self, obs, lnxi, lngamma):
        pass

    def decode(self, obs):
        """
        Get the most probable cluster id
        """
        z, lnp = self._eval_hidden_states(obs)
        return z.argmax(1)

    def simulate(self, T, mu, cv):
        n, d = mu.shape

        pi_cdf = np.exp(self._lnpi).cumsum()
        A_cdf = np.exp(self._lnA).cumsum(1)
        z = np.zeros(T, dtype=np.int)
        o = np.zeros((T, d))
        r = random(T)
        z[0] = (pi_cdf > r[0]).argmax()
        o[0] = sample_gaussian(mu[z[0]], cv[z[0]])
        for t in range(1, T):
            z[t] = (A_cdf[z[t - 1]] > r[t]).argmax()
            o[t] = sample_gaussian(mu[z[t]], cv[z[t]])
        return z, o
