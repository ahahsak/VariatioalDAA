import numpy as np
from numpy.random import rand, dirichlet, normal, random, randn
from scipy.cluster import vq
from scipy.special import gammaln, digamma
from scipy.linalg import eig, inv, cholesky
from vardaa.util import (logsum, log_like_gauss, kl_dirichlet,
                         kl_gauss_wishart, normalize, sample_gaussian,
                         e_lnpi_dirichlet)


class VbHmm():
    # VB-HMM with Gaussian emission probability.
    # VB-E step is Forward-Backward Algorithm.

    def __init__(self, n, obs, synthetic_T, synthetic_mu, synthetic_cv,
                 uPi0=0.5, uA0=0.5, m0=0, beta0=1, nu0=1, scale=10.0):
        # number of hidden states
        self.n_states = n

        # hyperparameters for prior
        # for initial prob
        self._upi = np.ones(n) * uPi0
        # for trans prob
        self._ua = np.ones((n, n)) * uA0
        # hyperparameters for emission distr(Gauss)
        self._m0 = m0
        self._beta0 = beta0
        # hyper parameters for emission distri(Wishert)
        self._nu0 = nu0

        # parameters for posterior
        # for initial prob
        self._wpi = np.array(self._upi)
        # for transition prob
        self._wa = np.array(self._ua)

        # log initial probability
        self._lnpi = np.log(np.tile(1.0 / n, n))
        # log transition probability matrix
        self._lnA = np.log(dirichlet([1.0] * n, n))

        if obs is None:
            self.codes, self.obs = simulate(
                synthetic_T, synthetic_mu, synthetic_cv, self._lnpi, self._lnA)
            obs = self.obs

        T, D = obs.shape

        # Initialize prior parameters
        if self._nu0 < D:
            self._nu0 += D
        self._m0 = np.mean(obs, 0)
        self._W0 = np.atleast_2d(np.cov(obs.T)) * scale

        # Initialize posterior parameters
        self.z = dirichlet(np.tile(1.0 / n, n), T)
        # mf parameters of emission distr (Gauss)
        self._m, _ = vq.kmeans2(obs, n, minit='points')
        self._beta = np.tile(self._beta0, n)
        # mf parameters of emission distr (Wishert)
        self._W = np.tile(np.array(self._W0), (n, 1, 1))
        self._nu = np.tile(float(T) / n, n)
        # auxiliary variable (PRML p.192 N_k S_k)
        self._s = np.array(self._W)

    def _allocate_fb(self, obs):
        T = len(obs)
        lnAlpha = np.zeros((T, self.n_states))  # log forward variable
        lnBeta = np.zeros((T, self.n_states))  # log backward variable
        lnXi = np.zeros((T - 1, self.n_states, self.n_states))
        return lnAlpha, lnBeta, lnXi

    def _forward(self, lnF, lnAlpha):
        # Use forward algorith to calculate forward variables and loglikelihood
        # input
        #  lnF [ndarray, shape (n,n_states)] : loglikelihood of emissions
        #  lnAlpha [ndarray, shape (n,n_states)] : log forward variable
        # output
        #  lnAlpha [ndarray, shape (n,n_states)] : log forward variable
        #  lnP [float] : lnP(X|theta)
        T = len(lnF)
        lnAlpha *= 0.0
        lnAlpha[0, :] = self._lnpi + lnF[0, :]

        for t in range(1, T):
            lnAlpha[t, :] = logsum(lnAlpha[t - 1, :] +
                                   self._lnA.T, 1) + lnF[t, :]

        return lnAlpha, logsum(lnAlpha[-1, :])

    def _backward(self, lnF, lnBeta):
        # backward algorithm to calculate backward variables and loglikelihood
        # input
        #    lnF [ndarray, shape (n,n_states)] : loglikelihood of emissions
        #    lnBeta [ndarray, shape (n,n_states)] : log backward variable
        # output
        #    lnBeta [ndarray, shape (n,n_states)] : log backward variable
        #    lnP [float] : lnP(X|theta)
        T = len(lnF)
        lnBeta[T - 1, :] = 0.0

        for t in range(T - 2, -1, -1):
            lnBeta[t, :] = logsum(
                self._lnA + lnF[t + 1, :] + lnBeta[t + 1, :], 1)

        return lnBeta, logsum(lnBeta[0, :] + lnF[0, :] + self._lnpi)

    def _log_like_f(self, obs):
        return log_like_gauss(obs, self._nu, self._W, self._beta, self._m)

    def _calculate_sufficient_statistics(self, obs, lnXi, lnGamma):
        # z[n,k] = Q(zn=k)
        nmix = self.n_states
        t, d = obs.shape
        self.z = np.exp(lnGamma)
        self._n = self.z.sum(0)
        self._xbar = np.dot(self.z.T, obs) / self._n[:, np.newaxis]
        for k in range(nmix):
            d_obs = obs - self._xbar[k]
            self._s[k] = np.dot((self.z[:, k] * d_obs.T), d_obs)

    def _update_parameters(self, obs, lnXi, lnGamma):
        nmix = self.n_states
        t, d = obs.shape
        # update parameters of initial prob
        self._wpi = self._upi + self.z[0]
        self._lnpi = e_lnpi_dirichlet(self._wpi)

        # update parameters of transition prob
        self._wa = self._ua + np.exp(lnXi).sum()
        self._lnA = digamma(self._wa) - digamma(self._wa)

        for k in range(nmix):
            self._lnA[k, :] = e_lnpi_dirichlet(self._wa[k, :])

        self._beta = self._beta0 + self._n
        self._nu = self._nu0 + self._n
        self._W = self._W0 + self._s

        for k in range(nmix):
            self._m[k] = (self._beta0 * self._m0 +
                          self._n[k] * self._xbar[k]) / self._beta[k]
            dx = self._xbar[k] - self._m0
            self._W[k] += (self._beta0 * self._n[k] /
                           self._beta[k] + self._n[k]) * np.outer(dx, dx)

    def _kl_div(self):
        # Compute KL divergence of initial and transition probabilities
        n_states = self.n_states
        kl_pi = kl_dirichlet(self._wpi, self._upi)
        kl_A = 0
        kl_g = 0
        kl = 0
        for k in range(n_states):
            kl_A += kl_dirichlet(self._wa[k], self._ua[k])
            kl_g += kl_gauss_wishart(self._nu[k], self._W[k], self._beta[k],
                                     self._m[k], self._nu0, self._W0,
                                     self._beta0, self._m0)
        kl += kl_pi + kl_A + kl_g
        return kl

    def _e_step(self, lnF, lnAlpha, lnBeta, lnXi):
        # lnF [ndarray, shape (n,n_states)] : loglikelihood of emissions
        # lnAlpha [ndarray, shape (n, n_states]: log forward message
        # lnBeta [ndarray, shape (n, n_states)]: log backward message
        # lnPx_f: log sum of p(x_n) by forward message for scalling
        # lnPx_b: log sum of p(x_n) by backward message for scalling

        T = len(lnF)
        # forward-backward algorithm
        lnAlpha, lnpx_f = self._forward(lnF, lnAlpha)
        lnBeta, lnpx_b = self._backward(lnF, lnBeta)
        # check if forward and backward were done correctly
        dlnp = lnpx_f - lnpx_b
        if abs(dlnp) > 1.0e-6:
            print("warning forward and backward are not equivalent")
        # compute lnXi for updating transition matrix
        lnXi = self._calculate_lnXi(lnXi, lnAlpha, lnBeta, lnF, lnpx_f)
        # compute lnGamma for postetior on hidden states
        lnGamma = lnAlpha + lnBeta - lnpx_f
        return lnXi, lnGamma, lnpx_f

    def _calculate_lnXi(self, lnXi, lnAlpha, lnBeta, lnF, lnpx_f):
        T = len(lnF)
        for i in range(self.n_states):
            for j in range(self.n_states):
                for t in range(T - 1):
                    lnXi[t, i, j] = lnAlpha[t, i] + self._lnA[i, j, ] + \
                        lnF[t + 1, j] + lnBeta[t + 1, j]
        lnXi -= lnpx_f
        return lnXi

    def _m_step(self, obs, lnXi, lnGamma):
        self._calculate_sufficient_statistics(obs, lnXi, lnGamma)
        self._update_parameters(obs, lnXi, lnGamma)

    def fit(self, obs, n_iter=10000, eps=1.0e-4,
            ifreq=10, old_f=1.0e20):
        # Fit the HMM via VB-EM algorithm
        old_f = 1.0e20
        lnAlpha, lnBeta, lnXi = self._allocate_fb(obs)

        for i in range(n_iter):
            # VB-E step
            lnF = self._log_like_f(obs)
            lnXi, lnGamma, lnp = self._e_step(lnF, lnAlpha, lnBeta, lnXi)
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
            # update parameters via VB-M step
            self._m_step(obs, lnXi, lnGamma)

    @staticmethod
    def simulate(T, mu, cv, lnpi, lnA):
        n, d = mu.shape
        pi_cdf = np.exp(lnpi).cumsum()
        A_cdf = np.exp(lnA).cumsum(1)
        z = np.zeros(T, dtype=np.int)
        o = np.zeros((T, d))
        r = random(T)
        z[0] = (pi_cdf > r[0]).argmax()
        o[0] = sample_gaussian(mu[z[0]], cv[z[0]])
        for t in range(1, T):
            z[t] = (A_cdf[z[t - 1]] > r[t]).argmax()
            o[t] = sample_gaussian(mu[z[t]], cv[z[t]])
        return z, o
