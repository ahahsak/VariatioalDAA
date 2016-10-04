import numpy as np
from numpy.random import rand, dirichlet, normal, random, randn
from scipy.cluster import vq
from scipy.special import gammaln, digamma
from scipy.linalg import eig, inv, cholesky
from util import *

"""
    VB-HMM with Gaussian emission probability.
    VB-E step is Forward-Backward Algorithm.
"""


class VB_HMM():

    def __init__(self, n, uPi0=0.5, uA0=0.5, m0=0.0,
                 beta0=1, nu0=1, s0=0.01):

        self.n_states = n
        self._lnPi = np.log(np.tile(1.0 / n, n))  # log initial probability
        # log transition probability
        self._lnA = np.log(dirichlet([1.0] * n, n))

        # 事前分布のハイパーパラメータ
        self._uPi = np.ones(n) * uPi0   # 初期状態確率
        self._uA = np.ones((n, n)) * uA0     # 遷移確率

        # 事後分布のパラメータ
        self._wPi = np.array(self._uPi)  # 初期確率
        self._wA = np.array(self._uA)  # 遷移確率

        self._m0 = m0
        self._beta0 = beta0
        self._nu0 = nu0
        self._s0 = s0

    def allocate_fb(self, obs):
        # fbアルゴリズムを走らせた時の一時保存用
        T = len(obs)
        lnAlpha = np.zeros((T, self.n_states))  # log forward variable
        lnBeta = np.zeros((T, self.n_states))  # log backward variable
        lnXi = np.zeros((T - 1, self.n_states, self.n_states))
        return lnAlpha, lnBeta, lnXi

    def forward(self, lnF, lnAlpha):
        """
        Use forward algorith to calculate forward variables and loglikelihood
        input
          lnF [ndarray, shape (n,n_states)] : loglikelihood of emissions
          lnAlpha [ndarray, shape (n,n_states)] : log forward variable
        output
          lnAlpha [ndarray, shape (n,n_states)] : log forward variable
          lnP [float] : lnP(X|theta)
        """
        T = len(lnF)
        lnAlpha *= 0.0
        lnAlpha[0, :] = self.lnPi + lnF[0, :]

        for t in range(1, T):
            lnAlpha[t, :] = logsum(lnAlpha[t - 1, :] +
                                   self.lnA.T, 1) + lnF[t, :]

        return lnAlpha, logsum(lnAlpha[-1, :])

    def backward(self, lnF, lnBeta):
        """
        Use backward algorith to calculate backward variables and loglikelihood
        input
            lnF [ndarray, shape (n,n_states)] : loglikelihood of emissions
            lnBeta [ndarray, shape (n,n_states)] : log backward variable
        output
            lnBeta [ndarray, shape (n,n_states)] : log backward variable
            lnP [float] : lnP(X|theta)
        """
        T = len(lnF)
        lnBeta[T - 1, :] = 0.0

        for t in range(T - 2, -1, -1):
            lnBeta[t, :] = logsum(
                self.lnA + lnF[t + 1, :] + lnBeta[t + 1, :], 1)

        return lnBeta, logsum(lnBeta[0, :] + lnF[0, :] + self.lnPi)

    def initialize_vbhmm(self, obs, scale=10.0):
        n_states = self.n_states

        T, D = obs.shape
        self.mu, temp = vq.kmeans2(obs, n_states)
        self.cv = np.tile(np.identity(D), (n_states, 1, 1))

        if self._nu0 < D:
            self._nu0 += D

        self._m0 = np.mean(obs, 0)
        self._V0 = np.atleast_2d(np.cov(obs.T)) * scale

        self.A = dirichlet([1.0] * n_states, n_states)   # A:状態遷移行列
        # self.lnA = np.log(A)

        self.pi = np.tile(1.0 / n_states, n_states)  # pi:初期状態確率
        # self.lnPi = np.log(pi)

        # posterior for hidden states
        self.z = dirichlet(np.tile(1.0 / n_states, n_states), T)
        # for mean vector
        self._m, temp = vq.kmeans2(obs, n_states, minit='points')
        self._beta = np.tile(self._beta0, n_states)
        # for covarience matrix
        self._V = np.tile(np.array(self._V0), (n_states, 1, 1))
        self._nu = np.tile(float(T) / n_states, n_states)

        # aux valable
        self._C = np.array(self._V)

    def _log_like_f(self, obs):
        return log_like_Gauss(obs, self._nu, self._V, self._beta, self._m)

    def _calculate_sufficient_statistics(self, obs, lnXi, lnGamma):
        # z[n,k] = Q(zn=k)
        nmix = self.n_states
        T, D = obs.shape
        self.z = np.exp(np.vstack(lnGamma))
        self.z0 = np.exp([lg[0] for lg in lnGamma]).sum(0)
        self._N = self.z.sum(0)
        self._Xbar = np.dot(self.z.T, obs) / self._N[:, np.newaxis]
        for k in range(nmix):
            d_obs = obs - self._Xbar[k]
            self._C = np.dot((self.z[:, k] * d_obs.T), d_obs)

    def _update_parameters(self, obs, lnXi, lnGamma):
        nmix = self.n_states
        T, D = obs.shape
        # update parameters of initial prob
        self._WPi = self._uPi + self.z0
        self._lnPi = digamma(self._wPi) - digamma(self._wPi.sum())

        # update parameters of transition prob
        self._wA = self._uA + np.exp(lnXi).sum()
        self._lnA = digamma(self._wA) - digamma(self._wA)

        for k in range(nmix):
            self._lnA[k, :] = digamma(
                self._wA[k, :]) - digamma(self._wA[k, :].sum())

        self._beta = self._beta0 + self._N
        self._nu = self._nu0 + self._N
        self._V = self._V0 + self._C
        for k in range(nmix):
            self._m[k] = (self._beta0 * self._m0 +
                          self._N[k] * self._Xbar[k]) / self._beta[k]
            dx = self._Xbar[k] - self._m0
            self._V[k] += (self._beta0 * self._N[k] /
                           self._beta[k]) * np.outer(dx, dx)

    def _KL_div(self):
        """
        Compute KL divergence of initial and transition probabilities
        """
        n_states = self.n_states
        KLPi = KL_Dirichlet(self._wPi, self._uPi)
        KLA = 0
        KLg = 0
        KL = 0
        for k in range(n_states):
            KLA += KL_Dirichlet(self._wA[k], self._uA[k])
            KLg += KL_GaussWishart(self._nu[k], self._V[k], self._beta[k],
                                   self._m[k], self._nu0, self._V0,
                                   self._beta0, self._m0)
        KL += KLPi + KLA + KLg
        return KL

    def get_expectations(self):
        """
        Calculate expectations of parameters over posterior distribution
        """
        self.A = self._wA / self._wA.sum(1)[:, np.newaxis]
        # <pi_k>_Q(pi_k)
        self.ev = eig(self.A.T)
        self.pi = normalize(np.abs(self.ev[1][:, self.ev[0].argmax()]))

        # <mu_k>_Q(mu_k,W_k)
        self.mu = np.array(self._m)

        # inv(<W_k>_Q(W_k))
        self.cv = self._V / self._nu[:, np.newaxis, np.newaxis]

        return self.pi, self.A, self.mu, self.cv

    def _Estep(self, lnF, lnAlpha, lnBeta, lnXi):
        """
        lnF [ndarray, shape (n,n_states)] : loglikelihood of emissions
        lnAlpha [ndarray, shape (n, n_states]: log forward message
        lnBeta [ndarray, shape (n, n_states)]: log backward message
        lnPx_f: log sum of p(x_n) by forward message for scalling
        lnPx_b: log sum of p(x_n) by backward message for scalling
        """
        T = len(lnF)
        # forward-backward algorithm
        lnAlpha, lnPx_f = self.forward(lnF, lnAlpha)
        lnBeta, lnPx_b = self.backward(lnF, lnBeta)

        # check if forward and backward were done correctly
        dlnP = lnPx_f - lnPx_b
        if abs(dlnP) > 1.0e-6:
            print("warning forward and backward are not equivalent")

        # compute lnXi for updating transition matrix
        for i in range(self.n_states):
            for j in range(self.n_states):
                for t in range(T - 1):
                    lnXi[t, i, j] = lnAlpha[t, i] + self.lnA[i, j, ] + \
                        lnF[t + 1, j] + lnBeta[t + 1, j]
        lnXi -= lnPx_f

        # compute lnGamma for postetior on hidden states
        lnGamma = lnAlpha + lnBeta - lnPx_f

        return lnXi, lnGamma, lnPx_f

    def Mstep(self, obs, lnXi, lnGamma):
        self._calculate_sufficient_statistics(obs, lnXi, lnGamma)
        self._update_parameters(obs, lnXi, lnGamma)

    def _eval_hidden_states(self, obs):
        """
        Performe one Estep.
        Then obtain variational free energy and posterior over hidden states
        """

        lnF = self._log_like_F(obs)
        lnAlpha, lnBeta, lnXi = self.allocate_fb(obs)
        lnXi, lnGamma, lnP = self._Estep(lnF, lnAlpha, lnBeta, lnXi)
        z = np.exp(lnGamma)
        return z, lnP

    def score(self, obs):
        """
        score the model
            input
              obs [ndarray, shape(nobs,ndim)] : observed data
            output
              F [float] : variational free energy of the model
        """
        n_obs = obs.shape
        z, lnP = self._eval_hidden_states(obs)
        F = -lnP + self._KL_div()
        return F

    def fit(self, obs, n_iter=10000, eps=1.0e-4,
            ifreq=10, old_F=1.0e20, init=True):
        '''Fit the HMM via VB-EM algorithm'''
        if init:
            self.initialize_vbhmm(obs)
            old_F = 1.0e20
            lnAlpha, lnBeta, lnXi = self.allocate_fb(obs)

        for i in range(n_iter):
            # VB-E step
            self.lnF = self._log_like_f(obs)
            self.lnXi, self.lnGamma, self.lnP = self._Estep(
                self.lnF, lnAlpha, lnBeta, lnXi)

            # check convergence
            KL = self._KL_div()
            F = -lnP + KL
            dF = F - old_F
            if(abs(dF) < eps):
                print("%8dth iter, Free Energy = %12.6e, dF = %12.6e" %
                      (i, F, dF))
                print("%12.6e < %12.6e Converged" % (dF, eps))
                break
            if i % ifreq == 0 and dF < 0.0:
                print("% 6dth iter, F = % 15.8e  df = % 15.8e" % (i, F, dF))
            elif dF >= 0.0:
                print("% 6dth iter, F = % 15.8e  df = % 15.8e warning" %
                      (i, F, dF))

            old_F = F
            print(old_F)

            # update parameters via VB-M step
            self.Mstep(obs, lnXi, lnGamma)

    def show_model(self, show_pi=True, show_A=True, show_mu=False,
                   show_cv=False, eps=1.0e-2):
        """
        return parameters of relavent clusters
        """
        self.get_expectations()
        ids = []
        sorted_ids = (-self.pi).argsort()
        for k in sorted_ids:
            if pi[k] > eps:
                ids.append(k)
        pi = self.pi[ids]
        mu = self.mu[ids]
        cv = self.cv[ids]
        A = np.array([AA[ids] for AA in self.A[ids]])
        for k in range(len(ids)):
            i = ids[k]
            print("\n%dth component, pi = %8.3g" % (k, pi[i]))
            print("cluster id =", i)
        if show_pi:
            print("pi = ", pi)
        if show_A:
            print("A = ", A)
        if show_mu:
            print("mu =", mu[i])
        if show_cv:
            print("cv =", cv[i])

        return ids, pi, A, mu, cv

    def decode(self, obs):
        """
        Get the most probable cluster id
        """
        z, lnP = self.eval_hidden_states(obs)
        return z.argmax(1)

    def simulate(self, T):
        N, D = self.mu.shape

        pi_cdf = np.exp(self._lnPi).cumsum()
        A_cdf = np.exp(self._lnA).cumsum(1)
        z = np.zeros(T, dtype=np.int)
        o = np.zeros((T, D))
        r = random(T)
        z[0] = (pi_cdf > r[0]).argmax()
        o[0] = sample_gaussian(self.mu[z[0]], self.cv[z[0]])
        for t in range(1, T):
            z[t] = (A_cdf[z[t - 1]] > r[t]).argmax()
            o[t] = sample_gaussian(self.mu[z[t]], self.cv[z[t]])
        return z, o
