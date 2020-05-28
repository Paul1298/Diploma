import sys
import time

import dadi
import moments
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm


# import momi

def dadi_model(n_pop, T, ns, pts):
    xx = dadi.Numerics.default_grid(pts)
    phi = dadi.PhiManip.phi_1D(xx)
    if n_pop >= 2:
        phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
        if n_pop == 2:
            T_loc = 2 * T
        elif n_pop == 3:
            T_loc = T
        phi = dadi.Integration.two_pops(phi, xx, T=T_loc, nu1=1.0, nu2=1.0)
    if n_pop == 3:
        phi = dadi.PhiManip.phi_2D_to_3D_split_2(xx, phi)
        phi = dadi.Integration.three_pops(phi, xx, T=T, nu1=1.0, nu2=1.0, nu3=1.0)
    return dadi.Spectrum.from_phi(phi, ns, [xx] * n_pop)


def moments_model(n_pop, T, ns):
    sts = moments.LinearSystem_1D.steady_state_1D(sum(ns))
    fs = moments.Spectrum(sts)
    if n_pop >= 2:
        x = 1 if n_pop > 2 else 4
        fs = moments.Manips.split_1D_to_2D(fs, ns[0], sum(ns[1:]))
        fs.integrate(Npop=[1.0, 1.0], tf=x * T)
    if n_pop >= 3:
        x = 1 if n_pop > 3 else 3
        fs = moments.Manips.split_2D_to_3D_2(fs, ns[1], sum(ns[2:]))
        fs.integrate(Npop=[1.0] * 3, tf=x * T)
    if n_pop >= 4:
        x = 1 if n_pop > 4 else 2
        fs = moments.Manips.split_3D_to_4D_3(fs, ns[2], sum(ns[3:]))
        fs.integrate(Npop=[1.0] * 4, tf=x * T)
    if n_pop >= 5:
        fs = moments.Manips.split_4D_to_5D_4(fs, ns[3], sum(ns[4:]))
        fs.integrate(Npop=[1.0] * 5, tf=T)
    return fs


# def momi_model(n_pop):
#     N=1e4
#     T = 3
#     T = 2 * N * T
#     model = momi.DemographicModel(N_e=N, gen_time=1,
#                               muts_per_gen=1.25e-8)
#
#     model.add_leaf("pop1", N=N)
#     if n_pop > 1:
#         model.add_leaf("pop2", N=N)
#         model.move_lineages("pop2", "pop1", t=8 * T)
#     if n_pop > 2:
#         model.add_leaf("pop3", N=N)
#         model.move_lineages("pop3", "pop2", t=7 * T)
#     if n_pop > 3:
#         model.add_leaf("pop4", N=N)
#         model.move_lineages("pop4", "pop3", t=6 * T)
#     if n_pop > 4:
#         model.add_leaf("pop5", N=N)
#         model.move_lineages("pop5", "pop4", t=5 * T)
#     if n_pop > 5:
#         model.add_leaf("pop6", N=N)
#         model.move_lineages("pop6", "pop5", t=4 * T)
#     if n_pop > 6:
#         model.add_leaf("pop7", N=N)
#         model.move_lineages("pop7", "pop6", t=3 * T)
#     if n_pop > 7:
#         model.add_leaf("pop8", N=N)
#         model.move_lineages("pop8", "pop7", t=2 * T)
#     if n_pop > 8:
#         model.add_leaf("pop9", N=N)
#         model.move_lineages("pop9", "pop8", t=T)
#     return model


def run_dadi_test(par, ns_1, pts_l):
    '''
Result:   
10 [0.002598285675048828, 0.0750267505645752, 1.0705108642578125]
20 [0.003746509552001953, 0.14794182777404785, 3.178598165512085]
30 [0.004683256149291992, 0.2380847930908203, 7.416461944580078]
40 [0.006814479827880859, 0.3744471073150635, 15.026614904403687]
50 [0.01135110855102539, 0.5303077697753906, 27.26491904258728]
60 [0.01256251335144043, 0.7062654495239258, 44.674824714660645]
70 [0.0161590576171875, 0.9219067096710205, 69.03412652015686]
80 [0.02075052261352539, 1.1637859344482422, 99.16849541664124]
    '''
    res = []
    for n_pop in range(1, 4):
        def func(T, ns, pts):
            return dadi_model(n_pop, T, ns, pts)

        t1 = time.time()
        func_ex = dadi.Numerics.make_extrap_log_func(func)
        model = func_ex(par, [ns_1] * n_pop, pts_l)
        t2 = time.time()
        res.append(t2 - t1)
    return res


def run_moments_test(par, ns_1):
    '''
Result:
10 [0.0015158653259277344, 0.015314579010009766, 0.11319804191589355, 1.2347056865692139, 23.203108072280884]
20 [0.0013666152954101562, 0.029877901077270508, 0.3729233741760254, 8.827671527862549, 355.97147393226624]
30 [0.001438140869140625, 0.05322885513305664, 0.8257763385772705, 31.57155466079712, 1374.6368079185486]
40 [0.0017266273498535156, 0.08972859382629395, 1.9885749816894531, 82.23191094398499, 4356.353564023972]
50 [0.0017697811126708984, 0.12981605529785156, 2.639880657196045, 179.8065574169159]
60 [0.0026514530181884766, 0.24073171615600586, 4.396363019943237, 318.94581270217896]
70 [0.0019762516021728516, 0.25220417976379395, 6.0970799922943115, 533.2001190185547]
80 [0.002245187759399414, 0.3430655002593994, 8.255565881729126, 969.6083562374115]
    '''
    res = []
    for n_pop in range(1, 6):
        def func(T, ns):
            return moments_model(n_pop, T, ns)

        t1 = time.time()
        model = func(par, [ns_1] * n_pop)
        t2 = time.time()
        t = t2 - t1
        print(t, file=sys.stderr)
        res.append(t)
    return res


# def run_momi_test(ns_1):
#     res = []
#     recoms_per_gen = 1.25e-8
#     bases_per_locus = int(5e5)
#     n_loci = 20
#     ploidy = 2
#     n_sample_dict = {}
#     for n_pop in range(1,10):
#         n_sample_dict['pop%d' % n_pop] = ns_1
#         model = momi_model(n_pop)
#         t1 = time.time()
# #        model._get_demo(n_sample_dict)
#         model.simulate_vcf(
#                     "outfile",
#                     recoms_per_gen=recoms_per_gen,
#                     length=bases_per_locus,
#                     ploidy=ploidy,
#                     sampled_n_dict=n_sample_dict,
#                     force=True)
#         t2 = time.time()
#         res.append(t2-t1)
#     return res

def run_test(mode):
    pts = [80, 90, 100]
    T = 3
    ns_list = [10, 20]#, 30, 40]#, 50, 60, 70, 80]
    # if mode == 'moments':
    #     ns_list = ns_list[4:]
    plt.figure(1)
    col_list = iter(cm.rainbow(np.linspace(0, 1, len(ns_list))))
    for ns, col in zip(ns_list, col_list):
        pts = [ns, ns + 10, ns + 20]
        if mode == 'dadi':
            times = run_dadi_test(T, ns, pts)
        if mode == 'moments':
            times = run_moments_test(T, ns)
        # if mode == 'momi':
        #     times = run_momi_test(ns)
        with open('log.txt', 'a') as f:
            print(ns, times, file=f)
        n_pop_pos = range(1, len(times) + 1)
        plt.plot(n_pop_pos, times, c=col, label=str(ns) + ' samples per population')
        plt.xticks(n_pop_pos, n_pop_pos)
        plt.xlabel('Число популяций')
        plt.ylabel('Время симуляции AFS (сек)')
    plt.yscale('log')
    # plt.title('Time complexity for %s simulations for different number of populations' % mode)
    plt.legend(loc=0)
    print('DONE')
    # plt.savefig('1.png')


def test_number_of_afs_entries():
    pts_l = [70, 80, 90]
    t1 = time.time()

    def func(T, ns, pts):
        return dadi_model(1, T, ns, pts)

    func_ex = dadi.Numerics.make_extrap_log_func(func)
    model = func_ex(3, [64], pts_l)
    t2 = time.time()
    print('Generation of 64 samples for 1 pop:\t %.5f sec.' % (t2 - t1))

    pts_l = [10, 20, 30]
    t1 = time.time()

    def func(T, ns, pts):
        return dadi_model(2, T, ns, pts)

    func_ex = dadi.Numerics.make_extrap_log_func(func)
    model = func_ex(3, [8, 8], pts_l)
    t2 = time.time()
    print('Generation of 8x8 samples for 2 pops:\t %.5f sec.' % (t2 - t1))

    pts_l = [10, 20, 30]
    t1 = time.time()

    def func(T, ns, pts):
        return dadi_model(3, T, ns, pts)

    func_ex = dadi.Numerics.make_extrap_log_func(func)
    model = func_ex(3, [4, 4, 4], pts_l)
    t2 = time.time()
    print('Generation of 4x4x4 samples for 3 pops:\t %.5f sec.' % (t2 - t1))


def plot_results(mode):
    if mode == 'dadi':
        res = {
            10: [0.002598285675048828, 0.0750267505645752, 1.0705108642578125],
            20: [0.003746509552001953, 0.14794182777404785, 3.178598165512085],
            30: [0.004683256149291992, 0.2380847930908203, 7.416461944580078],
            40: [0.006814479827880859, 0.3744471073150635, 15.026614904403687],
            50: [0.01135110855102539, 0.5303077697753906, 27.26491904258728],
            60: [0.01256251335144043, 0.7062654495239258, 44.674824714660645],
            70: [0.0161590576171875, 0.9219067096710205, 69.03412652015686],
            80: [0.02075052261352539, 1.1637859344482422, 99.16849541664124]
        }
    elif mode == 'moments':
        res = {
            10: [0.0015158653259277344, 0.015314579010009766, 0.11319804191589355, 1.2347056865692139,
                 23.203108072280884],
            20: [0.0013666152954101562, 0.029877901077270508, 0.3729233741760254, 8.827671527862549,
                 355.97147393226624],
            30: [0.001438140869140625, 0.05322885513305664, 0.8257763385772705, 31.57155466079712, 1374.6368079185486],
            40: [0.0017266273498535156, 0.08972859382629395, 1.9885749816894531, 82.23191094398499, 4356.353564023972],
            50: [0.0017697811126708984, 0.12981605529785156, 2.639880657196045, 179.8065574169159],
            60: [0.0026514530181884766, 0.24073171615600586, 4.396363019943237, 318.94581270217896],
            70: [0.0019762516021728516, 0.25220417976379395, 6.0970799922943115, 533.2001190185547],
            80: [0.002245187759399414, 0.3430655002593994, 8.255565881729126, 969.6083562374115]
        }
    elif mode == 'momi':
        res = {
            10: [0.040425777435302734, 0.2841489315032959, 0.5000772476196289, 0.7996480464935303, 1.1076393127441406,
                 1.4341752529144287, 1.835172414779663, 2.177875280380249, 2.4498701095581055],
            20: [0.07772159576416016, 0.449718713760376, 0.8464815616607666, 1.3791558742523193, 2.0268752574920654,
                 2.699354410171509, 3.6816837787628174, 4.350857734680176, 4.817714214324951],
            30: [0.05660057067871094, 0.5898139476776123, 1.2340788841247559, 2.057643413543701, 3.2617156505584717,
                 5.307035207748413, 5.32308292388916, 6.768655061721802, 7.576956033706665],
            40: [0.07962346076965332, 0.8049225807189941, 1.8476529121398926, 2.7939248085021973, 3.954272508621216,
                 5.470911264419556, 6.991794586181641, 8.649635553359985, 9.764487028121948],
            50: [0.09171724319458008, 1.1149280071258545, 2.055758476257324, 3.2204484939575195, 4.768917798995972,
                 6.351742506027222, 8.505664587020874, 9.884680032730103, 12.169224500656128],
            60: [0.09985589981079102, 1.248504400253296, 2.584294080734253, 4.832667350769043, 6.384091854095459,
                 8.91451621055603, 10.56020736694336, 13.327739953994751, 14.690232038497925],
            70: [0.12362122535705566, 1.304567575454712, 2.782144546508789, 4.662824630737305, 6.881324529647827,
                 8.63783073425293, 11.383946418762207, 13.8502197265625, 18.288761377334595],
            80: [0.1479353904724121, 1.61826491355896, 3.353036403656006, 5.353995323181152, 7.967868089675903,
                 10.720394134521484, 14.592302560806274, 18.05605435371399, 22.071446180343628]
        }

    plt.figure(figsize=(7, 6))
    res_keys = sorted(res)
    col_list = iter(cm.rainbow(np.linspace(0, 1, 8)))
    for ns, col in zip(res_keys, col_list):
        times = res[ns]
        n_pop_pos = range(1, len(times) + 1)
        plt.plot(n_pop_pos, times, c=col, label=str(ns) + ' samples per population')
    if mode == 'dadi':
        n_pop_pos = range(1, 4)
    elif mode == 'moments':
        n_pop_pos = range(1, 6)
    elif mode == 'momi':
        n_pop_pos = range(1, 10)
    plt.xticks(n_pop_pos, n_pop_pos)
    plt.xlabel('Number of populations')
    if mode == 'momi':
        plt.ylabel('Time of VCF simulation (sec)')
    else:
        plt.ylabel('Time of AFS simulation (sec)')

    plt.yscale('log')
    # plt.title('Time complexity for %s simulations for different number of populations' % mode)
    plt.legend(loc=0)
    print('DONE')
    plt.savefig('1.png')


# run_test('dadi')
run_test('moments')
# test_number_of_afs_entries()
# run_test('momi')
# plot_results('momi')
