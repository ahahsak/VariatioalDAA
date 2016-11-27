import os
import json
import numpy as np
from numpy.random import seed
import vardaa.hmm as hmm
from vardaa.hmm import VbHmm
import vardaa.models as models
from vardaa.models import Model
import vardaa.util as util
from nose.tools import assert_equal, ok_, nottest, assert_almost_equal

def test_show():
    model = get_expected_model()
    model.show()

def test_decode():
    model = VbHmm(3)
    mu, cv, lnA = create_parameters()
    model.lnA = lnA
    z, o2 = model.simulate(50 * 10, mu, cv)
    model.fit(o2)

    result = model.to_model()
    codes = result.decode(model.z)

def test_fit():
    obs = get_example_obs()
    expected = get_expected_model().to_dictionary()

    # fix seed for test
    seed(1)

    calculator = VbHmm(3)
    calculator.fit(obs)

    model = calculator.to_model()
    actuial = model.to_dictionary()

    assert_almost_equal(actuial, expected)

def get_example_obs():
    with open(os.path.dirname(__file__) + '/obs.json') as f:
        data = json.load(f)
        return np.array(data)

def get_expected_model():
    with open(os.path.dirname(__file__) + '/model.json') as f:
        data = json.load(f)
        return Model.from_dictionary(data)

@nottest
def test_getExpectations():
    pass

def create_parameters():
    mu = np.array([[3.0, 3.0], [0.0, 0.0], [-4.0, 0.0]])
    cv = np.tile(np.identity(2), (3, 1, 1))
    lnA = np.log([[0.9, 0.05, 0.05], [0.1, 0.7, 0.2], [0.1, 0.4, 0.5]])
    return mu, cv, lnA
