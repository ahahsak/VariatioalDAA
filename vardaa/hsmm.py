import numpy as np
from numpy import newaxis
from numpy.random import rand, dirichlet, normal, random, randn, gamma
from scipy.cluster import vq
from scipy.special import gammaln, digamma
from scipy.linalg import eig, inv, cholesky
from vardaa.util import (logsum, log_like_gauss, kl_dirichlet,
                         kl_gauss_wishart, normalize, sample_gaussian,
                         e_lnpi_dirichlet, log_like_poisson, kl_poisson_gamma)


class VbHsmm():
    """
    VB-HSMM with Gaussian emission and Poisson duration probability.
    VB-E step is Forward-Backward Algorithm.
    """

    def __init__(self, n, uPi0=0.5, uA0=0.5, m0=0.0,
                 beta0=0.25, nu0=4, a0=2 * 30, b0=2.0,
                 lambda0=None, mf_a0=None, mf_b0=None, trunc=None):

        self.n_states = n

        # log initial probability
        self._lnpi = np.log(np.tile(1.0 / n, n))
        # log transition probability
        self._lnA = np.log(dirichlet([1.0] * n, n))

        # prior parameter
        self._upi = np.ones(n) * uPi0   # first states prob
        self._ua = np.ones((n, n)) * uA0     # trans prob

        # posterior parameter
        self._wpi = np.array(self._upi)  # first states prob
        self._wa = np.array(self._ua)  # trans prob

        # Gauss-Wishert
        self._m0 = m0
        self._beta0 = beta0
        self._nu0 = nu0

        # Poisson-Gamma
        self._a0 = a0
        self._b0 = b0
        self._a = mf_a0 if mf_a0 is not None else self._a0
        self._b = mf_a0 if mf_b0 is not None else self._b0
        self._lambda0 = gamma(
            self._a0, 1 / self._b0) if lambda0 is None else lambda0
        self.trunc = trunc

    def _initialize_vbhsmm(self, obs, scale=10.0):
        n_states = self.n_states

        T, D = obs.shape
        self.mu, _ = vq.kmeans2(obs, n_states)
        self.cv = np.tile(np.identity(D), (n_states, 1, 1))

        if self._nu0 < D:
            self._nu0 += D

        self._m0 = np.mean(obs, 0)
        self._W0 = np.atleast_2d(np.cov(obs.T)) * scale

        self.A = dirichlet([1.0] * n_states, n_states)   # trans matrix
        self.pi = np.tile(1.0 / n_states, n_states)  # first state matrix

        # posterior for hidden states
        self.z = dirichlet(np.tile(1.0 / n_states, n_states), T)
        # Gauss
        self._m, _ = vq.kmeans2(obs, n_states, minit='points')
        self._beta = np.tile(self._beta0, n_states)
        # Wishert
        self._W = np.tile(np.array(self._W0), (n_states, 1, 1))
        self._nu = np.tile(float(T) / n_states, n_states)
        # auxiliary variable (PRML p.192 N_k S_k)
        self._s = np.array(self._W)

        # Poisson
        self._lambda = gamma(self._a, 1 / self._b)
        self._a, self._b = self._lambda * self._b0, self._b0

    def _allocate_fb(self, obs):
        T = len(obs)
        lnAlpha = np.zeros((T, self.n_states))  # log forward variable
        lnAlphastar = np.zeros((T, self.n_states))
        lnBeta = np.zeros((T, self.n_states))  # log backward variable
        lnBetastar = np.zeros((T, self.n_states))
        return lnAlpha, lnAlphastar, lnBeta, lnBetastar

    def _log_like_f(self, obs):
        lnEm = log_like_gauss(obs, self._nu, self._W, self._beta, self._m)
        lnDur = log_like_poisson(obs.shape[0], self.n_states, self._lambda)
        return lnEm, lnDur

    def _e_step(self, lnEm, lnDur, lnAlpha,
                lnAlphastar, lnBeta, lnBetastar):
        """
        lnEm [ndarray, shape (n,n_states)] : loglikelihood of emissions
        lnAlpha [ndarray, shape (n, n_states]: log forward message
        lnBeta [ndarray, shape (n, n_states)]: log backward message
        lnPx_f: log sum of p(x_n) by forward message for scalling
        lnPx_b: log sum of p(x_n) by backward message for scalling
        """
        # forward-backward algorithm
        lnAlpha, lnAlphastar, lnpx_f = self._forward(
            self._lnpi, self._lnA, lnEm, lnDur,
            lnAlpha, lnAlphastar, self.trunc)
        lnBeta, lnBetastar, lnpx_b = self._backward(
            self._lnpi, self._lnA, lnEm, lnDur,
            lnBeta, lnBetastar, self.trunc)
        # T = len(lnEm)
        # lnXi = np.zeros((T - 1, self.n_states, self.n_states))
        # check if forward and backward were done correctly
        # dlnp = lnpx_f - lnpx_b
        # if abs(dlnp) > 1.0e-6:
        # print("warning forward and backward are not equivalent")
        # compute lnXi for updating transition matrix
        lnXi = self.posterior_transitions(self.n_states, lnAlpha,
                                          lnBeta, self._lnA, lnpx_f)
        # compute lnGamma for postetior on hidden states
        lnGamma = lnAlpha + lnBeta - lnpx_f
        # compute lnDpost for postetior on duration
        lnDpost = self.posterior_durations(
            lnAlphastar, lnBeta, lnEm, lnDur, lnpx_f)
        return lnXi, lnGamma, lnDpost, lnpx_f

    def _m_step(self, obs, lnXi, lnGamma, lnDpost):
        self._calculate_sufficient_statistics(obs, lnGamma, lnXi, lnDpost)
        self._update_parameters()

    def _calculate_sufficient_statistics(self, obs, lnGamma, lnXi, lnDpost):
        # transitions
        self.tr = np.exp(lnXi).sum(0)
        self.counts = np.atleast_2d(self.tr).sum(0)

        # emmitions
        self.z = np.exp(lnGamma)
        # N_k in PRML(10.51)
        self._n = self.z.sum(0)
        # \bar{x}_k in PRML(10.52)
        self._xbar = np.dot(self.z.T, obs) / self._n[:, newaxis]
        for k in range(self.n_states):
            d_obs = obs - self._xbar[k]
            # S_k in PRML(10.53)
            self._s[k] = np.dot((self.z[:, k] * d_obs.T), d_obs)
        # durations
        self.d = np.exp(lnDpost.T)
        data = [np.arange(1, self.d[s].shape[0] + 1)
                for s in range(self.n_states)]
        weight = [self.d[s] for s in range(self.n_states)]
        self._nd = sum(w.sum() for w in weight)
        self._totd = sum(w.dot(d) for w, d in zip(weight, data))

    def _update_parameters(self):
        nmix = self.n_states
        # update parameters of initial prob
        self._wpi = self._upi + self.z[0]
        self._lnpi = e_lnpi_dirichlet(self._wpi)

        # update parameters of transition prob
        self._wa = self._ua + self.counts
        self._weights = self._wa / self._wa.sum()  # for plotting
        self._lnA = digamma(self._wa) - digamma(self._wa)
        for k in range(nmix):
            self._lnA[k, :] = e_lnpi_dirichlet(self._wa[k, :])

        # update parameters of emmition distr
        self._beta = self._beta0 + self._n
        self._nu = self._nu0 + self._n
        self._W = self._W0 + self._s
        for k in range(nmix):
            self._m[k] = (self._beta0 * self._m0 +
                          self._n[k] * self._xbar[k]) / self._beta[k]
            dx = self._xbar[k] - self._m0
            self._W[k] += (self._beta0 * self._n[k] /
                           self._beta[k] + self._n[k]) * np.outer(dx, dx)

        # update parameters of duration distribution
        self._a = self._a0 + self._nd
        self._b = self._b0 + self._totd
        self._lambda = self._a / self._b

    def fit(self, obs, n_iter=10000, eps=1.0e-4,
            ifreq=10, old_f=1.0e20):
        '''Fit the HSMM via VB-EM algorithm'''
        self._initialize_vbhsmm(obs)
        old_f = 1.0e20
        (lnAlpha, lnAlphastar,
            lnBeta, lnBetastar) = self._allocate_fb(obs)

        for i in range(n_iter):
            # VB-E step
            lnEm, lnDur = self._log_like_f(obs)
            lnXi, lnGamma, lnDpost, lnp = self._e_step(
                lnEm, lnDur, lnAlpha, lnAlphastar, lnBeta,
                lnBetastar)
            # update parameters via VB-M step
            self._m_step(obs, lnXi, lnGamma, lnDpost)
            # check convergence
            kl = self._kl_div()
            f = -lnp + kl
            df = f - old_f
            if(abs(df) < eps):
                print("%8dth iter, Free Energy = %12.6e, dF = %12.6e" %
                      (i, f, df))
                print("%12.6e < %12.6e Converged" % (df, eps))
                break
            if i % ifreq == 0 and df < 0.0:
                print("% 6dth iter, F = % 15.8e  df = % 15.8e" % (i, f, df))
            elif df >= 0.0:
                print("% 6dth iter, F = % 15.8e  df = % 15.8e warning" %
                      (i, f, df))
            old_f = f
            print(old_f)

    def _kl_div(self):
        """
        Compute KL divergence of initial and transition probabilities
        """
        n_states = self.n_states
        kl_pi = kl_dirichlet(self._wpi, self._upi)
        kl_A = 0
        kl_g = 0
        kl_p = 0
        kl = 0
        for k in range(n_states):
            kl_A += kl_dirichlet(self._wa[k], self._ua[k])
            kl_g += kl_gauss_wishart(self._nu[k], self._W[k], self._beta[k],
                                     self._m[k], self._nu0, self._W0,
                                     self._beta0, self._m0)
        kl_p += kl_poisson_gamma(self._lambda, self._a, self._b,
                                 self._lambda0, self._a0, self._b0)
        kl += kl_pi + kl_A + kl_g + kl_p
        return kl

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

    @staticmethod
    def _forward(lnpi, lnA, lnEm, lnDur,
                 lnAlpha, lnAlphastar, trunc=None):
        """
        Use forward algorith to calculate forward variables and loglikelihood
        input
          lnEm [ndarray, shape (n,n_states)] : loglikelihood of emissions
          lnAlpha [ndarray, shape (n,n_states)] : log forward variable
        output
          lnAlpha [ndarray, shape (n,n_states)] : log forward variable
          lnP [float] : lnP(X|theta)(normalizer)
        """
        T = len(lnEm)
        D = trunc if trunc is not None else T
        lnAlphastar[0] = lnpi
        for t in range(T - 1):
            dmax = min(D, t + 1)
            a = lnAlphastar[t + 1 - dmax:t + 1] + lnDur[:dmax][::-1] + \
                np.cumsum(lnEm[t + 1 - dmax:t + 1][::-1], axis=0)[::-1]
            lnAlpha[t] = logsum(a, axis=0)

            a = lnAlpha[t][:, newaxis] + lnA
            lnAlphastar[t + 1] = logsum(a, axis=0)
        t = T - 1
        dmax = min(D, t + 1)
        a = lnAlphastar[t + 1 - dmax:t + 1] + lnDur[:dmax][::-1] + \
            np.cumsum(lnEm[t + 1 - dmax:t + 1][::-1], axis=0)[::-1]
        lnAlpha[t] = logsum(a, axis=0)
        lnP = logsum(lnAlpha[-1:])
        return lnAlpha, lnAlphastar, lnP

    @staticmethod
    def _backward(lnpi, lnA, lnEm, lnDur, lnBeta, lnBetastar, trunc=None):
        """
        Use backward algorith to calculate backward variables and loglikelihood
        input
            lnEm [ndarray, shape (n,n_states)] : loglikelihood of emissions
            lnBeta [ndarray, shape (n,n_states)] : log backward variable
        output
            lnBeta [ndarray, shape (n,n_states)] : log backward variable
            lnP [float] : lnP(X|theta)(normalizer)
        """
        T = len(lnEm)
        D = trunc if trunc is not None else T
        lnBeta[T - 1] = 0.
        for t in reversed(range(T - 1)):
            # TODO: right-censoring
            dmax = min(D, T - t)
            b = lnBeta[t:t + dmax] + lnDur[:dmax] + \
                np.cumsum(lnEm[t:t + dmax], axis=0)
            lnBetastar[t] = np.logaddexp.reduce(b, axis=0)
            if dmax < D:
                lnBetastar[t] = np.logaddexp(
                    lnBetastar[t], np.logaddexp.reduce(
                        lnDur[dmax:], axis=0) + np.sum(lnEm[t:], axis=0))
            if t > 0:
                b = lnBetastar[t] + lnA
                lnBeta[t - 1] = np.logaddexp.reduce(b, axis=1)
        lnBeta[T - 1] = 0.0
        lnP = logsum(lnBetastar[0, :] + lnpi)
        return lnBeta, lnBetastar, lnP

    @staticmethod
    def posterior_transitions(n_states, lnAlpha, lnBetastar, lnA, normalizer):
        lntrans = lnAlpha[:-1, :, newaxis] + lnBetastar[1:, newaxis, :] +\
            lnA[np.newaxis, ...]
        lntrans -= normalizer
        return lntrans

    @staticmethod
    def posterior_durations(lnAlphastar, lnBeta, lnEm, lnDur, lnpx_f):
        # mattj's thesis (5.2.23)
        T = len(lnEm)
        logpmfs = -np.inf * np.ones_like(lnAlphastar)
        for t in range(T):
            lnEm_cum = np.cumsum(lnEm[t:T], axis=0)
            np.logaddexp(lnDur[:T - t] + lnAlphastar[t] + lnBeta[t:] +
                         lnEm_cum - lnpx_f,
                         logpmfs[:T - t], out=logpmfs[:T - t])
        return logpmfs
