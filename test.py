import numpy as np
from hmm import VB_HMM


model = VB_HMM(3)
model.mu = np.array([[3.0, 3.0], [0.0, 0.0], [-4.0, 0.0]])
model.cv = np.tile(np.identity(2), (3, 1, 1))
model.lnA = np.log([[0.9, 0.05, 0.05], [0.1, 0.7, 0.2], [0.1, 0.4, 0.5]])
z, o2 = model.simulate(50 * 10)
model.fit(o2)
z2 = model.decode(o2)
model.show_model(True, True, True, True)
