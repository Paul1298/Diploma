import moments
import numpy as np


def model_func(params, ns):
    nu1, nu2, nu3, m12, m13, m23, T1, T2 = params
    m112 = 0
    sts = moments.LinearSystem_1D.steady_state_1D(sum(ns))
    fs = moments.Spectrum(sts)

    fs = moments.Manips.split_1D_to_2D(fs, ns[0], ns[1] + ns[2])

    fs.integrate(Npop=[nu1, nu2], tf=T1, m=np.array([[0, m112], [m112, 0]]))

    fs = moments.Manips.split_2D_to_3D_2(fs, ns[1], ns[2])

    fs.integrate(Npop=[nu1, nu2, nu3], tf=T2, m=np.array([[0, m12, m13], [m12, 0, m23], [m13, m23, 0]]))

    return fs


p_ids = ['n', 'n', 'n', 'm', 'm', 'm', 't', 't']


lower_bound = [0.01, 0.01, 0.01, 0, 0, 0, 0, 0]


upper_bound = [100, 100, 100, 10, 10, 10, 5, 5]
