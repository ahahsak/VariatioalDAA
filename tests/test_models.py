import numpy as np
import vardaa.hmm as hmm
from vardaa.hmm import VbHmm
import vardaa.models as models
from vardaa.models import Model
import vardaa.util as util
from nose.tools import assert_equal, ok_, nottest


@nottest
def test_show():
    model = VbHmm(3)
    mu = np.array([[3.0, 3.0], [0.0, 0.0], [-4.0, 0.0]])
    cv = np.tile(np.identity(2), (3, 1, 1))
    # model.lnA = np.log([[0.9, 0.05, 0.05], [0.1, 0.7, 0.2], [0.1, 0.4, 0.5]])
    z, o2 = model.simulate(50 * 10, mu, cv)
    model.fit(o2)
    result = Model(model.pi, model.A, model.mu, model.cv,
                   model._wa, model._nu, model._v, model._m)
    # z2 = model.decode(o2)
    # model.show_model(True, True, True, True)
    result.show()


@nottest
def test_decode():
    model = VbHmm(3)
    mu = np.array([[3.0, 3.0], [0.0, 0.0], [-4.0, 0.0]])
    cv = np.tile(np.identity(2), (3, 1, 1))
    # model.lnA = np.log([[0.9, 0.05, 0.05], [0.1, 0.7, 0.2], [0.1, 0.4, 0.5]])
    z, o2 = model.simulate(50 * 10, mu, cv)
    model.fit(o2)
    result = Model(model.pi, model.A, model.mu, model.cv,
                   model._wa, model._nu, model._v, model._m)
    codes = result.decode(model.z)
    pass


@nottest
def test_score():
    pass


@nottest
def test_getExpectations():
    pass
