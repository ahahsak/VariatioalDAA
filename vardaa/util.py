import numpy as np
from scipy import stats
from scipy.special import gammaln, digamma
from scipy.linalg import eig, det, solve, inv, cholesky
from scipy.spatial.distance import cdist
from numpy.random import randn


def logsum(A, axis=None):
    """
    Computes the sum of A assuming A is in the log domain.
    Returns log(sum(exp(A), axis)) while minimizing the possibility of
    over/underflow.
    """
    Amax = A.max(axis)
    if axis and A.ndim > 1:
        shape = list(A.shape)
        shape[axis] = 1
        Amax.shape = shape
    Asum = np.log(np.sum(np.exp(A - Amax), axis))
    Asum += Amax.reshape(Asum.shape)
    if axis:
        # Look out for underflow.
        Asum[np.isnan(Asum)] = - np.Inf
    return Asum


def normalize(A, axis=None):
    A += np.finfo(float).eps
    Asum = A.sum(axis)
    if axis and A.ndim > 1:
        # Make sure we don't divide by zero.
        Asum[Asum == 0] = 1
        shape = list(A.shape)
        shape[axis] = 1
        Asum.shape = shape
    return A / Asum


def kl_dirichlet(alpha1, alpha2):
    """
    KL-div of Dirichlet distribution KL[q(alpha1)||p(alpha2)]
    input
      alpha1 [ndarray, shape (nmix)] : parameter of 1st Dirichlet dist
      alpha2 [ndarray, shape (nmix)] : parameter of 2nd Dirichlet dist
    """

    kl = - lnz_dirichlet(alpha1) + lnz_dirichlet(alpha2) \
        + np.dot((alpha1 - alpha2), (digamma(alpha1) - digamma(alpha1.sum())))

    return kl


def lnz_dirichlet(alpha):
    """
    log normalization constant of Dirichlet distribution
    input
      alpha [ndarray, shape (nmix)] : parameter of Dirichlet distribution
    """

    Z = gammaln(alpha).sum() - gammaln(alpha.sum())
    return Z


def lnz_wishart(nu, W):
    """
    log normalization constant of Wishart distribution
    input
      nu [float] : dof parameter of Wichart distribution
      V [ndarray, shape (D x D)] : base matrix of Wishart distribution
      note <CovMat> = V/nu
    """

    # if nu < len(V) + 1:
    #     raise ValueError("dof parameter nu must larger than len(V)")

    D = len(W)
    lnZ = 0.5 * nu * (D * np.log(2.0) - np.log(det(W))) \
        + gammaln(np.arange(nu + 1 - D, nu + 1) * 0.5).sum()

    return lnZ


def log_like_gauss(obs, nu, W, beta, m):
    """
    Log probability for Gaussian with full covariance matrices.
    Here mean vectors and covarience matrices are probability variable with
    respect to Gauss-Wishart distribution.
    """
    nobs, ndim = obs.shape
    nmix = len(m)
    lnF = np.empty((nobs, nmix))
    for k in range(nmix):
        dln2pi = ndim * np.log(2.0 * np.pi)
        lndetW = - e_lndetw_wishart(nu[k], W[k])
        cv = W[k] / nu[k]
        q = _sym_quad_form(obs, m[k], cv) + ndim / beta[k]
        lnF[:, k] = -0.5 * (dln2pi + lndetW + q)
    return lnF


def log_like_poisson(obs, lmbda):
    return stats.poisson.logpmf(obs, lmbda)


def _sym_quad_form(x, mu, A):
    """
    calculate x.T * inv(A) * x
    """
    q = (cdist(x, mu[np.newaxis], "mahalanobis", VI=inv(A))**2).reshape(-1)
    return q


def e_lndetw_wishart(nu, W):
    """
    mean of log determinant of precision matrix over Wishart <lndet(W)>
    input
      nu [float] : dof parameter of Wichart distribution
      W [ndarray, shape (D x D)] : base matrix of Wishart distribution
    """

    D = len(W)
    E = D * np.log(2.0) - np.log(det(W)) + \
        digamma(np.arange(nu + 1 - D, nu + 1) * 0.5).sum()

    return E


def e_lnpi_dirichlet(alpha):
    return digamma(alpha) - digamma(alpha.sum())


def kl_wishart(nu1, W1, nu2, W2):
    """
    KL-div of Wishart distribution KL[q(nu1,W1)||p(nu2,W2)]
    """

    # if nu1 < len(W1) + 1:
    #     raise ValueError("dof parameter nu1 must larger than len(W1)")
    # if nu2 < len(V2) + 1:
    #     raise ValueError("dof parameter nu2 must larger than len(W2)")

    # if len(W1) != len(W2):
    #     raise ValueError("dimension of two matrix dont match, %d and %d" % (
    #         len(W1), len(W2)))

    D = len(W1)
    kl = 0.5 * ((nu1 - nu2) * e_lndetw_wishart(nu1, W1) + nu1 *
                (np.trace(solve(W1, W2)) - D)) - lnz_wishart(nu1, W1) + lnz_wishart(nu2, W2)
    return kl


def kl_gauss_wishart(nu1, W1, beta1, m1, nu2, W2, beta2, m2):
    """
    KL-div of Gauss-Wishart distr KL[q(nu1,W1,beta1,m1)||p(nu2,W2,beta2,m2)
    """
    if len(m1) != len(m2):
        raise ValueError(
            "dimension of two mean dont match, %d and %d" % (len(m1), len(m2)))

    D = len(m1)

    # first assign KL of Wishart
    kl1 = kl_wishart(nu1, W1, nu2, W2)

    # the rest terms
    kl2 = 0.5 * (D * (np.log(beta1 / float(beta2)) + beta2 / float(beta1) -
                      1.0) + beta2 * nu1 * np.dot((m1 - m2),
                                                  solve(W1, (m1 - m2))))
    kl = kl1 + kl2
    return kl


def sample_gaussian(m, cv, n=1):
    """Generate random samples from a Gaussian distribution.
    Parameters
    ----------
    m : array, shape (ndim)
        Mean of the distribution.
    cv : array, shape (ndim,ndim)
        Covariance of the distribution.
    n : int, optional
        Number of samples to generate. Defaults to 1.
    Returns
    -------
    obs : array, shape (ndim, n)
        Randomly generated sample
    """

    ndim = len(m)
    r = randn(n, ndim)
    if n == 1:
        r.shape = (ndim,)

    cv_chol = cholesky(cv)
    r = np.dot(r, cv_chol.T) + m

    return r
