import numpy as np
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


def KL_Dirichlet(alpha1, alpha2):
    """
    KL-div of Dirichlet distribution KL[q(alpha1)||p(alpha2)]
    input
      alpha1 [ndarray, shape (nmix)] : parameter of 1st Dirichlet dist
      alpha2 [ndarray, shape (nmix)] : parameter of 2nd Dirichlet dist
    """

    KL = - lnZ_Dirichlet(alpha1) + lnZ_Dirichlet(alpha2) \
        + np.dot((alpha1 - alpha2), (digamma(alpha1) - digamma(alpha1.sum())))

    return KL


def lnZ_Dirichlet(alpha):
    """
    log normalization constant of Dirichlet distribution
    input
      alpha [ndarray, shape (nmix)] : parameter of Dirichlet dist
    """

    Z = gammaln(alpha).sum() - gammaln(alpha.sum())
    return Z


def lnZ_Wishart(nu, V):
    """
    log normalization constant of Wishart distribution
    input
      nu [float] : dof parameter of Wichart distribution
      V [ndarray, shape (D x D)] : base matrix of Wishart distribution
      note <CovMat> = V/nu
    """
    '''
    if nu < len(V) + 1:
        raise ValueError("dof parameter nu must larger than len(V)")
    '''
    D = len(V)
    lnZ = 0.5 * nu * (D * np.log(2.0) - np.log(det(V))) \
        + gammaln(np.arange(nu + 1 - D, nu + 1) * 0.5).sum()

    return lnZ


def log_like_Gauss(obs, nu, V, beta, m):
    """
    Log probability for Gaussian with full covariance matrices.
    Here mean vectors and covarience matrices are probability variable with
    respect to Gauss-Wishart distribution.
    """
    nobs, ndim = obs.shape
    nmix = len(m)
    lnf = np.empty((nobs, nmix))
    for k in range(nmix):
        dln2pi = ndim * np.log(2.0 * np.pi)
        lndetV = - E_lndetW_Wishart(nu[k], V[k])
        cv = V[k] / nu[k]
        q = _sym_quad_form(obs, m[k], cv) + ndim / beta[k]
        lnf[:, k] = -0.5 * (dln2pi + lndetV + q)

    return lnf


def _sym_quad_form(x, mu, A):
    """
    calculate x.T * inv(A) * x
    """
    q = (cdist(x, mu[np.newaxis], "mahalanobis", VI=inv(A))**2).reshape(-1)
    return q


def E_lndetW_Wishart(nu, V):
    """
    mean of log determinant of precision matrix over Wishart <lndet(W)>
    input
      nu [float] : dof parameter of Wichart distribution
      V [ndarray, shape (D x D)] : base matrix of Wishart distribution
    """

    D = len(V)
    E = D * np.log(2.0) - np.log(det(V)) + \
        digamma(np.arange(nu + 1 - D, nu + 1) * 0.5).sum()

    return E


def KL_Wishart(nu1, V1, nu2, V2):
    """
    KL-div of Wishart distribution KL[q(nu1,V1)||p(nu2,V2)]
    """
    '''
    if nu1 < len(V1) + 1:
        raise ValueError("dof parameter nu1 must larger than len(V1)")
    if nu2 < len(V2) + 1:
        raise ValueError("dof parameter nu2 must larger than len(V2)")

    if len(V1) != len(V2):
        raise ValueError("dimension of two matrix dont match, %d and %d" % (
            len(V1), len(V2)))
    '''

    D = len(V1)
    KL = 0.5 * ((nu1 - nu2) * E_lndetW_Wishart(nu1, V1) + nu1 *
                (np.trace(solve(V1, V2)) - D)) - lnZ_Wishart(nu1, V1)
    + lnZ_Wishart(nu2, V2)

    '''
    if KL < _small_negative_number:
        print(nu1, nu2, V1, V2)
        raise ValueError("KL must be larger than 0")
    '''
    return KL


def KL_GaussWishart(nu1, V1, beta1, m1, nu2, V2, beta2, m2):
    """
    KL-div of Gauss-Wishart distr KL[q(nu1,V1,beta1,m1)||p(nu2,V2,beta2,m2)
    """
    if len(m1) != len(m2):
        raise ValueError(
            "dimension of two mean dont match, %d and %d" % (len(m1), len(m2)))

    D = len(m1)

    # first assign KL of Wishart
    KL1 = KL_Wishart(nu1, V1, nu2, V2)

    # the rest terms
    KL2 = 0.5 * (D * (np.log(beta1 / float(beta2)) + beta2 / float(beta1) -
                      1.0) + beta2 * nu1 * np.dot((m1 - m2),
                                                  solve(V1, (m1 - m2))))

    KL = KL1 + KL2
    '''
    if KL < _small_negative_number:
        raise ValueError("KL must be larger than 0")
    '''
    return KL


def sample_gaussian(m, cv, n=1):
    """Generate random samples from a Gaussian distribution.
    Parameters
    ----------
    m : array_like, shape (ndim)
        Mean of the distribution.
    cv : array_like, shape (ndim,ndim)
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
