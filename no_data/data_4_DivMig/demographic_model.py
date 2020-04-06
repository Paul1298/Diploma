import moments
import numpy as np


def model_func(params, ns):
    nu1, nu234, nu2, nu34, nu3, nu4, m1_234, m1_2, m1_34, m2_34, m1_3, m1_4, m2_3, m2_4, m3_4, T1, T2, T3 = params
    sts = moments.LinearSystem_1D.steady_state_1D(sum(ns))
    fs = moments.Spectrum(sts)

    fs = moments.Manips.split_1D_to_2D(fs, ns[0], sum(ns[1:]))
    m = np.array([
        [0, m1_234],
        [m1_234, 0]
    ])
    fs.integrate(Npop=[nu1, nu234], tf=T1, m=m)

    fs = moments.Manips.split_2D_to_3D_2(fs, ns[1], sum(ns[2:]))
    m = np.array([
        [0, m1_2, m1_34],
        [m1_2, 0, m2_34],
        [m1_34, m2_34, 0]
    ])
    fs.integrate(Npop=[nu1, nu2, nu34], tf=T2, m=m)

    fs = moments.Manips.split_3D_to_4D_3(fs, ns[2], sum(ns[3:]))
    m = np.array([
        [0, m1_2, m1_3, m1_4],
        [m1_2, 0, m2_3, m2_4],
        [m1_3, m2_3, 0, m3_4],
        [m1_4, m2_4, m3_4, 0]
    ])
    fs.integrate(Npop=[nu1, nu2, nu3, nu4], tf=T3, m=m)

    return fs


p_ids = ['n', 'n', 'n', 'n', 'n', 'n', 'm', 'm', 'm', 'm', 'm', 'm', 'm', 'm', 'm', 't', 't', 't']

lower_bound = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

upper_bound = [100, 100, 100, 100, 100, 100, 10, 10, 10, 10, 10, 10, 10, 10, 10, 5, 5, 5]

max_possible_ll = -62920.7645870188
