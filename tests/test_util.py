import numpy
import vbhmm.util as util
from nose.tools import assert_equal, ok_, nottest

@nottest
def test_logsum():
    pass

@nottest
def test_normalize():
    pass

@nottest
def test_KL_Dirichlet():
    pass

def test_lnZ_Dirichlet():
    upi = numpy.ones(10) * 0.5
    wpi = numpy.array(upi)
    actual = util.lnZ_Dirichlet(wpi)
    ok_(2.54 < actual < 2.55)

@nottest
def test_lnZ_Wishart():
    pass

@nottest
def test_log_like_Gauss():
    pass

@nottest
def test__sym_quad_form():
    pass

@nottest
def test_E_lndetW_Wishart():
    pass

@nottest
def test_KL_Wishart():
    pass

@nottest
def test_KL_GaussWishart():
    pass

@nottest
def test_sample_gaussian():
    pass
