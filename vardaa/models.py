import numpy as np
from vardaa.hmm import VbHmm
from vardaa.util import normalize
from scipy.linalg import eig


class HMM(VbHmm):

    def __init__(self, n):
        VbHmm.__init__(self, n)

    def fit_vb_art(self, T, mu, cv):
        states_art, obs = VbHmm.generate_obs_gauss(self, T, mu, cv)
        self._vb_em(obs)

    def _vb_em(self, obs, n_iter=10000, eps=1.0e-4,
               ifreq=10, old_f=1.0e20, init=True):
        return VbHmm.fit(self, obs, n_iter, eps, ifreq, old_f, init)

    def show(self, eps=1.0e-2):
        """
        return parameters of relavent clusters
        """
        self._get_expectations(self.pi, self.A, self.mu, self.cv)
        ids = []
        sorted_ids = (-self.pi).argsort()
        for k in sorted_ids:
            if self.pi[k] > eps:
                ids.append(k)
        pi = self.pi[ids]
        mu = self.mu[ids]
        cv = self.cv[ids]
        A = np.array([AA[ids] for AA in self.A[ids]])
        '''
        for k in range(len(ids)):
            i = ids[k]
            print("\n%dth component, pi = %8.3g" % (k, pi[i]))
            print("cluster id =", i)
        '''
        return ids, pi, A, mu, cv

    def decode(self, obs):
        """
        Get the most probable cluster id
        """
        z, lnp = VbHmm._eval_hidden_states(self, obs)
        return z.argmax(1)

    def score(self, obs):
        """
        score the model
            input
              obs [ndarray, shape(nobs,ndim)] : observed data
            output
              F [float] : variational free energy of the model
        """
        n_obs = obs.shape
        z, lnp = VbHmm._eval_hidden_states(obs)
        f = -lnp + VbHmm._kl_div()
        return f

    def _get_expectations(self, pi, A, mu, cv):
        """
        Calculate expectations of parameters over posterior distribution
        """
        A = self._wa / self._wa.sum(1)[:, np.newaxis]
        # <pi_k>_Q(pi_k)
        ev = eig(A.T)
        pi = normalize(np.abs(ev[1][:, ev[0].argmax()]))

        # <mu_k>_Q(mu_k,W_k)
        mu = np.array(self._m)

        # inv(<W_k>_Q(W_k))
        cv = self._v / self._nu[:, np.newaxis, np.newaxis]

        return pi, A, mu, cv

    def plot():
        pass
