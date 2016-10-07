# extend vbhmm to vb for word model
from vbhmm.hmm import VbHmm
import numpy as np


class VbWordModel(VbHmm):

    def __init__(self):
        VbHmm.__init__(self, N)
