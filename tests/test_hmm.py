import numpy as np
import vbhmm.hmm as hmm
from vbhmm.hmm import VB_HMM
import vbhmm.util as util
from nose.tools import assert_equal, ok_, nottest


@nottest
def test_allocate_fb():
    pass


@nottest
def test_forward():
    pass


@nottest
def test_backward():
    pass


@nottest
def test_initialize_vbhmm():
    pass


@nottest
def test__log_like_f():
    pass


@nottest
def test__calculate_sufficient_statistics():
    pass


@nottest
def test__update_parameters():
    pass


@nottest
def test__KL_div():
    pass


@nottest
def test_getExpectations():
    pass


@nottest
def test__Estep():
    pass


@nottest
def test_Mstep():
    pass


@nottest
def test__eval_hidden_states():
    pass


@nottest
def test_score():
    pass


@nottest
def test_fit():
    pass


@nottest
def test_showModel():
    model = VB_HMM(3)
    model.mu = np.array([[3.0, 3.0], [0.0, 0.0], [-4.0, 0.0]])
    model.cv = np.tile(np.identity(2), (3, 1, 1))
    model.lnA = np.log([[0.9, 0.05, 0.05], [0.1, 0.7, 0.2], [0.1, 0.4, 0.5]])
    z, o2 = model.simulate(50 * 10)
    model.fit(o2)
    z2 = model.decode(o2)
    model.show_model(True, True, True, True)


@nottest
def test_decode():
    pass


@nottest
def test_simulate():
    pass
