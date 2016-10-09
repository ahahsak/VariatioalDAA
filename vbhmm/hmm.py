import numpy as np
from numpy.random import rand, dirichlet, normal, random, randn
from scipy.cluster import vq
from scipy.special import gammaln, digamma
from scipy.linalg import eig, inv, cholesky
from vbhmm.util import *

"""
    VB-HMM with Gaussian emission probability.
    VB-E step is Forward-Backward Algorithm.
"""


class VbHmm():

    def __init__(self, N, UPI0=0.5, UA0=0.5, M0=0.0,
                 BETA0=1, NU0=1, S0=0.01):

        self.n_states = N
        # log initial probability
        self._lnpi = np.log(np.tile(1.0 / N, N))
        # log transition probability
        self._lnA = np.log(dirichlet([1.0] * N, N))

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

    def _initialize_vbhmm(self, obs, scale=10.0):
        n_states = self.n_states

        T, D = obs.shape
        self.mu, temp = vq.kmeans2(obs, n_states)
        self.cv = np.tile(np.identity(D), (n_states, 1, 1))

        if self._nu0 < D:
            self._nu0 += D

        self._m0 = np.mean(obs, 0)
        self._v0 = np.atleast_2d(np.cov(obs.T)) * scale

        self.A = dirichlet([1.0] * n_states, n_states)   # A:状態遷移行列
        self.pi = np.tile(1.0 / n_states, n_states)  # pi:初期状態確率

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

    def _log_like_f(self, obs):
        return log_like_gauss(obs, self._nu, self._v, self._beta, self._m)

    def _calculate_sufficient_statistics(self, obs, lnxi, lngamma):
        # z[n,k] = Q(zn=k)
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

    def _eval_hidden_states(self, obs):
        """
        Performe one Estep.
        Then obtain variational free energy and posterior over hidden states
        """

        lnf = self._log_like_f(obs)
        lnalpha, lnbeta, lnxi = self._allocate_fb(obs)
        lnxi, lngamma, lnp = self._e_step(lnf, lnalpha, lnbeta, lnxi)
        z = np.exp(lngamma)
        return z, lnp

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
