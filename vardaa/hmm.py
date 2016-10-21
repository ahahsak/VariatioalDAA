import numpy as np
from numpy.random import rand, dirichlet, normal, random, randn
from scipy.cluster import vq
from scipy.special import gammaln, digamma
from scipy.linalg import eig, inv, cholesky
from vardaa.util import logsum, log_like_gauss, kl_dirichlet, kl_gauss_wishart, normalize, sample_gaussian


class VbHmm():
    """
    VB-HMM with Gaussian emission probability.
    VB-E step is Forward-Backward Algorithm.
    """

    def __init__(self, n, uPi0=0.5, uA0=0.5, m0=0.0, beta0=1, nu0=1, s0=0.01):

        self.n_states = n
        # log initial probability
        self._lnpi = np.log(np.tile(1.0 / n, n))
        # log transition probability
        self._lnA = np.log(dirichlet([1.0] * n, n))

        # 事前分布のハイパーパラメータ
        self._upi = np.ones(n) * uPi0   # 初期状態確率
        self._ua = np.ones((n, n)) * uA0     # 遷移確率

        # 事後分布のパラメータ
        self._wpi = np.array(self._upi)  # 初期確率
        self._wa = np.array(self._ua)  # 遷移確率

        self._m0 = m0
        self._beta0 = beta0
        self._nu0 = nu0
        self._s0 = s0

    def _allocate_fb(self, obs):
        # fbアルゴリズムを走らせた時の一時保存用
        T = len(obs)
        lnAlpha = np.zeros((T, self.n_states))  # log forward variable
        lnBeta = np.zeros((T, self.n_states))  # log backward variable
        lnXi = np.zeros((T - 1, self.n_states, self.n_states))
        return lnAlpha, lnBeta, lnXi

    def _forward(self, lnF, lnAlpha):
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
        lnAlpha[0, :] = self._lnpi + lnF[0, :]

        for t in range(1, T):
            lnAlpha[t, :] = logsum(lnAlpha[t - 1, :] +
                                   self._lnA.T, 1) + lnF[t, :]

        return lnAlpha, logsum(lnAlpha[-1, :])

    def _backward(self, lnF, lnBeta):
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
                self._lnA + lnF[t + 1, :] + lnBeta[t + 1, :], 1)

        return lnBeta, logsum(lnBeta[0, :] + lnF[0, :] + self._lnpi)

    def _initialize_vbhmm(self, obs, scale=10.0):
        n_states = self.n_states

        T, D = obs.shape
        self.mu, _ = vq.kmeans2(obs, n_states)
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
        self._m, _ = vq.kmeans2(obs, n_states, minit='points')
        self._beta = np.tile(self._beta0, n_states)
        # for covarience matrix
        self._v = np.tile(np.array(self._v0), (n_states, 1, 1))
        self._nu = np.tile(float(T) / n_states, n_states)

        # aux valable
        self._c = np.array(self._v)

    def _log_like_f(self, obs):
        return log_like_gauss(obs, self._nu, self._v, self._beta, self._m)

    def _calculate_sufficient_statistics(self, obs, lnXi, lnGamma):
        # z[n,k] = Q(zn=k)
        nmix = self.n_states
        t, d = obs.shape
        self.z = np.exp(np.vstack(lnGamma))
        self.z0 = np.exp([lg[0] for lg in lnGamma]).sum(0)
        self._n = self.z.sum(0)
        self._xbar = np.dot(self.z.T, obs) / self._n[:, np.newaxis]
        for k in range(nmix):
            d_obs = obs - self._xbar[k]
            self._c[k] = np.dot((self.z[:, k] * d_obs.T), d_obs)

    def _update_parameters(self, obs, lnXi, lnGamma):
        nmix = self.n_states
        t, d = obs.shape
        # update parameters of initial prob
        self._wpi = self._upi + self.z0
        self._lnpi = digamma(self._wpi) - digamma(self._wpi.sum())

        # update parameters of transition prob
        self._wa = self._ua + np.exp(lnXi).sum()
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

    def _e_step(self, lnF, lnAlpha, lnBeta, lnXi):
        """
        lnF [ndarray, shape (n,n_states)] : loglikelihood of emissions
        lnAlpha [ndarray, shape (n, n_states]: log forward message
        lnBeta [ndarray, shape (n, n_states)]: log backward message
        lnPx_f: log sum of p(x_n) by forward message for scalling
        lnPx_b: log sum of p(x_n) by backward message for scalling
        """
        T = len(lnF)
        # forward-backward algorithm
        lnAlpha, lnpx_f = self._forward(lnF, lnAlpha)
        lnBeta, lnpx_b = self._backward(lnF, lnBeta)

        # check if forward and backward were done correctly
        dlnp = lnpx_f - lnpx_b
        if abs(dlnp) > 1.0e-6:
            print("warning forward and backward are not equivalent")

        # compute lnXi for updating transition matrix
        for i in range(self.n_states):
            for j in range(self.n_states):
                for t in range(T - 1):
                    lnXi[t, i, j] = lnAlpha[t, i] + self._lnA[i, j, ] + \
                        lnF[t + 1, j] + lnBeta[t + 1, j]
        lnXi -= lnpx_f

        # compute lnGamma for postetior on hidden states
        lnGamma = lnAlpha + lnBeta - lnpx_f

        return lnXi, lnGamma, lnpx_f

    def _m_step(self, obs, lnXi, lnGamma):
        self._calculate_sufficient_statistics(obs, lnXi, lnGamma)
        self._update_parameters(obs, lnXi, lnGamma)

    def fit(self, obs, n_iter=10000, eps=1.0e-4,
            ifreq=10, old_f=1.0e20, init=True):
        '''Fit the HMM via VB-EM algorithm'''
        if init:
            self._initialize_vbhmm(obs)
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

    '''
    def _eval_hidden_states(self, obs):
        """
        Performe one Estep.
        Then obtain variational free energy and posterior over hidden states
        """

        lnF = self._log_like_f(obs)
        lnAlpha, lnBeta, lnXi = self._allocate_fb(obs)
        lnXi, lnGamma, lnp = self._e_step(lnF, lnAlpha, lnBeta, lnXi)
        z = np.exp(lnGamma)
        return z, lnp
    '''
