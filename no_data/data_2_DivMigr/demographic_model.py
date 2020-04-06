import moments
import numpy as np


def model_func(params, ns):
    nu1, nu2, m12, m21, T = params
    sts = moments.LinearSystem_1D.steady_state_1D(sum(ns))
    fs = moments.Spectrum(sts)

    fs = moments.Manips.split_1D_to_2D(fs, ns[0], ns[1])

    fs.integrate(Npop=[nu1, nu2], tf=T, m=np.array([[0, m12], [m21, 0]]))
    return fs


p_ids = ['n', 'n', 'm', 'm', 't']

lower_bound = [0.01, 0.01, 0, 0, 0]

upper_bound = [100, 100, 10, 10, 5]

max_possible_ll = -1310.930985309239
