# extend vbhmm to vb for word model
from vbhmm.hmmbase import GaussianHmmBase
import numpy as np


class VbWordModel(GaussianHmmBase):

    def __init__(self, N):
        GaussianHmmBase.__init__(self, N)
