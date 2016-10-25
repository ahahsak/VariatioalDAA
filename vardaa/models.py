import numpy as np
from vardaa.hmm import VbHmm
from vardaa.util import normalize
from scipy.linalg import eig


class Model():

    def __init__(self, wa, nu, v, m):
        pi, A, mu, cv = Model._get_expectations(wa, nu, v, m)

        self.pi = pi
        self.A = A
        self.mu = mu
        self.cv = cv

    def show(self, eps=1.0e-2):
        """
        return parameters of relavent clusters
        """
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

    def decode(self, z):
        """
        Get the most probable cluster id
        """
        return z.argmax(1)

    '''
    def score(self, obs):
        """
        score the model
            input
              obs [ndarray, shape(nobs,ndim)] : observed data
            output
              F [float] : variational free energy of the model
        """
        # n_obs = obs.shape
        z, lnp = self._eval_hidden_states(obs)
        f = -lnp + VbHmm._kl_div()
        return f
    '''

    @staticmethod
    def _get_expectations(wa, nu, v, m):
        """
        Calculate expectations of parameters over posterior distribution
        """
        A = wa / wa.sum(1)[:, np.newaxis]
        # <pi_k>_Q(pi_k)
        ev = eig(A.T)
        pi = normalize(np.abs(ev[1][:, ev[0].argmax()]))

        # <mu_k>_Q(mu_k,W_k)
        mu = np.array(m)

        # inv(<W_k>_Q(W_k))
        cv = v / nu[:, np.newaxis, np.newaxis]

        return pi, A, mu, cv
