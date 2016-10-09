import numpy as np
from numpy.random import rand, dirichlet, normal, random, randn
from scipy.cluster import vq
from scipy.special import gammaln, digamma
from scipy.linalg import eig, inv, cholesky
from vardaa.hmmbase import GaussianHmmBase
from vardaa.util import *

"""
    VB-HMM with Gaussian emission probability.
    VB-E step is Forward-Backward Algorithm.
"""


class VbHmm(GaussianHmmBase):

    def __init__(self, N, UPI0=0.5, UA0=0.5, M0=0.0,
                 BETA0=1, NU0=1, S0=0.01):
        GaussianHmmBase.__init__(self, N)

        # 事前分布のハイパーパラメータ
        self._upi = np.ones(N) * UPI0   # 初期状態確率
        self._ua = np.ones((N, N)) * UA0     # 遷移確率

        # 事後分布のパラメータ
        self._wpi = np.array(self._upi)  # 初期確率
        self._wa = np.array(self._ua)  # 遷移確率

        self._m0 = M0
        self._beta0 = BETA0
        self._nu0 = NU0
        self._s0 = S0

    def _log_like_f(self, obs):
        return log_like_gauss(obs, self._nu, self._v, self._beta, self._m)

    def _initialize_vbhmm(self, obs, scale=10.0):
        GaussianHmmBase._initialize_hmm(self, obs)

        n_states = self.n_states
        T, D = obs.shape

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

    def _calculate_sufficient_statistics(self, obs, lnxi, lngamma):
        # z[n,k] = q(zn=k)
        nmix = self.n_states
        t, d = obs.shape
        self.z = np.exp(np.vstack(lngamma))
        self.z0 = np.exp([lg[0] for lg in lngamma]).sum(0)
        self._n = self.z.sum(0)
        self._xbar = np.dot(self.z.T, obs) / self._n[:, np.newaxis]
        for k in range(nmix):
            d_obs = obs - self._xbar[k]
            self._c[k] = np.dot((self.z[:, k] * d_obs.T), d_obs)

    def _update_parameters(self, obs, lnxi, lngamma):
        nmix = self.n_states
        t, d = obs.shape
        # update parameters of initial prob
        self._wpi = self._upi + self.z0
        self._lnpi = digamma(self._wpi) - digamma(self._wpi.sum())

        # update parameters of transition prob
        self._wa = self._ua + np.exp(lnxi).sum()
        self._lnA = digamma(self._wa) - digamma(self._wa)
        for k in range(nmix):
            self._lnA[k, :] = digamma(
                self._wa[k, :]) - digamma(self._wa[k, :].sum())

        # update parameters of emmition prob (Gaussian distribution)
        self._beta = self._beta0 + self._n
        self._nu = self._nu0 + self._n
        self._v = self._v0 + self._c

        for k in range(nmix):
            self._m[k] = (self._beta0 * self._m0 +
                          self._n[k] * self._xbar[k]) / self._beta[k]
            dx = self._xbar[k] - self._m0
            self._v[k] += (self._beta0 * self._n[k] /
                           self._beta[k] + self._n[k]) * np.outer(dx, dx)

    def _kl_div(self):
        """
        Compute KL divergence of initial and transition probabilities
        """
        n_states = self.n_states
        kl_pi = kl_dirichlet(self._wpi, self._upi)
        kl_A = 0
        kl_g = 0
        kl = 0
        for k in range(n_states):
            kl_A += kl_dirichlet(self._wa[k], self._ua[k])
            kl_g += kl_gauss_wishart(self._nu[k], self._v[k], self._beta[k],
                                     self._m[k], self._nu0, self._v0,
                                     self._beta0, self._m0)
        kl += kl_pi + kl_A + kl_g
        return kl

    def _get_expectations(self):
        """
        Calculate expectations of parameters over posterior distribution
        """
        self.A = self._wa / self._wa.sum(1)[:, np.newaxis]
        # <pi_k>_Q(pi_k)
        self.ev = eig(self.A.T)
        self.pi = normalize(np.abs(self.ev[1][:, self.ev[0].argmax()]))

        # <mu_k>_Q(mu_k,W_k)
        self.mu = np.array(self._m)

        # inv(<W_k>_Q(W_k))
        self.cv = self._v / self._nu[:, np.newaxis, np.newaxis]

        return self.pi, self.A, self.mu, self.cv

    def score(self, obs):
        """
        score the model
            input
              obs [ndarray, shape(nobs,ndim)] : observed data
            output
              F [float] : variational free energy of the model
        """
        n_obs = obs.shape
        z, lnp = self._eval_hidden_states(obs)
        f = -lnp + self._kl_div()
        return f

    def fit(self, obs, N_ITER=10000, EPS=1.0e-4,
            IFREQ=10, OLD_F=1.0e20, INIT=True):
        '''Fit the HMM via VB-EM algorithm'''
        if INIT:
            self._initialize_vbhmm(obs)
            OLD_F = 1.0e20
            lnalpha, lnbeta, lnxi = self._allocate_fb(obs)

        for i in range(N_ITER):
            # VB-E step
            lnf = self._log_like_f(obs)
            lnxi, lngamma, lnp = self._e_step(lnf, lnalpha, lnbeta, lnxi)

            # check convergence
            kl = self._kl_div()
            f = -lnp + kl
            df = f - OLD_F
            if(abs(df) < EPS):
                print("%8dth iter, Free Energy = %12.6e, dF = %12.6e" %
                      (i, f, df))
                print("%12.6e < %12.6e Converged" % (df, EPS))
                break
            if i % IFREQ == 0 and df < 0.0:
                print("% 6dth iter, F = % 15.8e  df = % 15.8e" % (i, f, df))
            elif df >= 0.0:
                print("% 6dth iter, F = % 15.8e  df = % 15.8e warning" %
                      (i, f, df))

            OLD_F = f
            print(OLD_F)

            # update parameters via VB-M step
            self._m_step(obs, lnxi, lngamma)

    def show_model(self, SHOW_PI=True, SHOW_A=True, SHOW_MU=False,
                   SHOW_CV=False, EPS=1.0e-2):
        """
        return parameters of relavent clusters
        """
        self._get_expectations()
        ids = []
        sorted_ids = (-self.pi).argsort()
        for k in sorted_ids:
            if self.pi[k] > EPS:
                ids.append(k)
        pi = self.pi[ids]
        mu = self.mu[ids]
        cv = self.cv[ids]
        A = np.array([AA[ids] for AA in self.A[ids]])
        for k in range(len(ids)):
            i = ids[k]
            print("\n%dth component, pi = %8.3g" % (k, pi[i]))
            print("cluster id =", i)
        if SHOW_PI:
            print("pi = ", pi)
        if SHOW_A:
            print("A = ", A)
        if SHOW_MU:
            print("mu =", mu[i])
        if SHOW_CV:
            print("cv =", cv[i])

        return ids, pi, A, mu, cv
