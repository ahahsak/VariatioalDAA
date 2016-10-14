import numpy as np
import vardaa.hmm as hmm
from vardaa.hmm import VbHmm
import vardaa.models as models
from vardaa.models import HMM
import vardaa.util as util
from nose.tools import assert_equal, ok_, nottest


@nottest
def fit_vb_art():
    model = HMM(3)
    mu = np.array([[3.0, 3.0], [0.0, 0.0], [-4.0, 0.0]])
    cv = np.tile(np.identity(2), (3, 1, 1))
    model.fit_vb_art(50 * 10, mu, cv)


@nottest
def test_show():
    model = HMM(3)
    mu = np.array([[3.0, 3.0], [0.0, 0.0], [-4.0, 0.0]])
    cv = np.tile(np.identity(2), (3, 1, 1))
    state_seq, obs_data = VbHmm.generate_obs_gauss(50 * 10, mu, cv)
    model._vb_em(obs_data)
    model.show()


@nottest
def test_decode():
    model = HMM(3)
    mu = np.array([[3.0, 3.0], [0.0, 0.0], [-4.0, 0.0]])
    cv = np.tile(np.identity(2), (3, 1, 1))
    state_seq, obs_data = VbHmm.generate_obs_gauss(50 * 10, mu, cv)
    model.decode(obs_data)


@nottest
def test_score():
    pass


@nottest
def test_getExpectations():
    pass
