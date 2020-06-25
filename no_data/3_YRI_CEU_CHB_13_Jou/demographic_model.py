import moments
import numpy


def model_func(params, ns):
    nuAf, nuB, nuEu0, nuEu, nuAs0, nuAs, mAfB, mAfEu, mAfAs, mEuAs, TAf, TB, TEuAs = params
    n1, n2, n3 = ns
    theta = 0.37976
    sts = moments.LinearSystem_1D.steady_state_1D(n1 + n2 + n3, theta=theta)
    fs = moments.Spectrum(sts)

    fs.integrate([nuAf], TAf, 0.05, theta=theta)

    fs = moments.Manips.split_1D_to_2D(fs, n1, n2 + n3)

    mig1 = numpy.array([[0, mAfB],
                        [mAfB, 0]])
    fs.integrate([nuAf, nuB], TB, 0.05, m=mig1, theta=theta)

    fs = moments.Manips.split_2D_to_3D_2(fs, n2, n3)

    nuEu_func = lambda t: nuEu0 * (nuEu / nuEu0) ** (t / TEuAs)
    nuAs_func = lambda t: nuAs0 * (nuAs / nuAs0) ** (t / TEuAs)
    nu2 = lambda t: [nuAf, nuEu_func(t), nuAs_func(t)]
    mig2 = numpy.array([[0, mAfEu, mAfAs],
                        [mAfEu, 0, mEuAs],
                        [mAfAs, mEuAs, 0]])

    fs.integrate(nu2, TEuAs, 0.05, m=mig2, theta=theta)

    return fs


max_possible_ll = -20001.302062

p_ids = ['n', 'n', 'n', 'n', 'n', 'n', 'm', 'm', 'm', 'm', 't', 't', 't']

lower_bound = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0, 0, 0, 0, 0, 0, 0]

upper_bound = [100, 100, 100, 100, 100, 100, 10, 10, 10, 10, 5, 5, 5]
