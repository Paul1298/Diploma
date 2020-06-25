import importlib
import os

import matplotlib.pyplot as plt
import moments
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MaxNLocator

# {'C0',   'C1',     'C2',    'C3',  'C4',     'C5',    'C6',   'C7',   'C8',    'C9'}
# {'blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'}
color_map = {
    'BayesOpt (MPI)': 3,
    'BayesOpt (EI)': 5,
    'BayesOpt + GADMA': 9,

    'GADMA': 2,
    'Random search': 0
}


def get_dir(data_dir, date=None):
    if 'results' not in os.listdir(data_dir):
        return

    results_dir = os.path.join(data_dir, 'results')
    if date is None:
        last_mo = os.path.join(results_dir, sorted(os.listdir(results_dir))[-1])
        last_day = os.path.join(last_mo, sorted(os.listdir(last_mo))[-1])

        last_start_dir = os.path.join(last_day,
                                      sorted(os.listdir(last_day))[-1])
    else:
        last_start_dir = os.path.join(results_dir, date)

    dirnames = next(os.walk(last_start_dir))[1]
    return last_start_dir, dirnames


def get_max_ll(model_file):
    dem_model = importlib.import_module(model_file.replace('/', '.').rstrip('.py'))
    return dem_model.__getattribute__('max_possible_ll')


def get_params(params_file):
    name = params_file.replace('/', '.').rstrip('.py')
    dem_model = importlib.import_module(name)
    return dem_model.__getattribute__('num_init_pts')


def draw_model(data_dir, path_dir, algorithm, best_model_params, best_ll):
    model_file = os.path.join(data_dir, 'demographic_model.py')

    dem_model = importlib.import_module(model_file.replace(os.path.sep, '.').rstrip('.py'))

    N_A = 11293

    # Draw model
    samples = 10
    model = moments.ModelPlot.generate_model(dem_model.model_func, best_model_params,
                                             (samples, samples, samples, samples, samples))

    save_file = os.path.join(path_dir, f'{algorithm}_model.png')
    fig_bg_color = '#ffffff'
    fig_kwargs = {'figsize': (3.4, 2.6), 'dpi': 200, 'facecolor': fig_bg_color,
                  'edgecolor': fig_bg_color}
    fig, ax = plt.subplots(**fig_kwargs)
    moments.ModelPlot.plot_model(model,
                                 ax=ax,
                                 save_file=save_file,
                                 # fig_title='',
                                 fig_title=fr'{algorithm}: {best_ll:.2f}',
                                 # fig_title=fr'{algorithm}: $\log(\mathcal{{L}})$ = {best_ll:.2f}',
                                 # pop_labels=['YRI', 'CEU', 'CHB', 'JPT'],
                                 # pop_labels=['Pop1', 'Pop2', 'Pop3', 'Pop4', 'Pop5'],
                                 # nref=N_A,
                                 nref=10000,
                                 draw_scale=False,
                                 # gen_time=0.029,
                                 gen_time=1,
                                 # gen_time_units="Thousand years",
                                 gen_time_units="Generations",
                                 grid=False,
                                 tick_size=9,
                                 reverse_timeline=True)

    text_color = '#002b36'
    ax.set_title(fr'{algorithm}: {best_ll:.2f}', color=text_color, fontsize=16)
    # ax.set_ylabel('')
    ax.set_xlabel('Time Ago (Generations)', color=text_color, fontsize=12)

    fig.savefig(save_file, bbox_inches='tight')


def draw(data_dir, date):
    last_start_dir, dirnames = get_dir(data_dir, date)
    dirnames = filter(lambda x: not x.startswith('_'), dirnames)
    # a.remove('random_search')
    # a.remove('gadma')
    dirnames = sorted(dirnames)
    # dirnames = ['GADMA']  # , 'BayesOpt']  # , 'BayesOpt + GADMA']
    # dirnames = ['BayesOpt']  # , 'BayesOpt + GADMA']

    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=200)
    # fig, ax = plt.subplots(figsize=(9.6, 5.4), dpi=200)

    max_possible_ll = int(get_max_ll(os.path.join(data_dir, 'demographic_model.py')))
    # print(max_possible_ll)

    # xmin = 0.16,
    # print(max_possible_ll)

    ax.axhline(max_possible_ll, color='r', linestyle='--', linewidth=1, label=r'max possible $\log(\mathcal{{L}})$')

    min_iter_num = 0
    shift = 0
    starts = 0
    init_pts = 16
    shifted = False
    min_first_ll = 0

    spec_map = {}

    for_anim = []
    lines = []
    for i, algorithm in enumerate(sorted(dirnames)):
        col = f'C{color_map.get(algorithm, min(filter(lambda x: x + 1 not in color_map.values(), color_map.values())) + 1)}'
        color_map[algorithm] = int(col[1:])

        # if algorithm == 'my_bayes(EI)':
        #    continue
        lls = []
        local_lls = []
        iter_num = 0
        cur_min_iter_num = np.inf
        q = 0
        algo_dir = os.path.join(last_start_dir,
                                algorithm)
        for dirpath, _, files in os.walk(algo_dir):
            start_num = dirpath.split('/')[-1].lstrip('start')
            # if start_num != '0':
            #     continue
            for f in files:
                log_file = os.path.join(dirpath, f)

                if f == 'evaluations.log':
                    q += 1
                    # if q >= 4:
                    #     break
                    with open(log_file) as file:
                        next(file)

                        cur_lls = []
                        iter_num = 0
                        best_ll = -np.inf
                        cur_max_ll = -np.inf
                        prevLL = 0
                        stop_check = True
                        for iter_num, line in enumerate(file):
                            if iter_num == 256:
                                break
                            parts = line.strip().split('\t')

                            logLL = -float(parts[1])

                            # if stop_check and logLL == prevLL:
                            #     stop_check = False
                            #     print(dirpath, iter_num)
                            # prevLL = logLL

                            cur_max_ll = max(cur_max_ll, logLL)
                            cur_lls.append(cur_max_ll)

                            if logLL > best_ll:
                                best_ll = logLL
                                best_model_params = np.fromstring(parts[2][1:-1], dtype=float, sep=',')
                                best_ll_iter_num = iter_num

                        # ax.plot(best_ll_iter_num, best_ll, '.', mew=3, color=col)
                        # ax.text(best_ll_iter_num - 1, best_ll, start_num, color=col)

                        if iter_num > 0:
                            lls.append(cur_lls)
                            cur_min_iter_num = min(cur_min_iter_num, iter_num - 1)
                elif f == 'params.py':
                    init_pts = get_params(log_file)
                elif f == 'local.log':
                    cur_lls = []
                    best_loc_ll = -np.inf
                    cur_max_ll = -np.inf
                    with open(log_file) as file:
                        for line in file:
                            if line.startswith('Beginning optimization'):
                                line = next(file)
                                iter_num = 1
                                while not line.startswith('Finished optimization'):
                                    parts = ''.join(' '.join(line.strip().split()).split(',')).split()
                                    # print(parts)
                                    # break
                                    logLL = float(parts[2])

                                    cur_max_ll = max(cur_max_ll, logLL)
                                    cur_lls.append(cur_max_ll)

                                    if logLL > best_loc_ll:
                                        best_loc_ll = logLL
                                        # best_model_params = np.fromstring(parts[2][1:-1], dtype=float, sep=',')
                                        loc_iter = iter_num
                                    line = next(file)
                                    iter_num += 1
                    if iter_num > 0:
                        local_lls.append(cur_lls)
                    # print(loc_iter, best_loc_ll, best_loc_ll - best_ll)
                    # plt.arrow(best_ll_iter_num, best_ll, loc_iter, best_loc_ll - best_ll, color=col, ls=':', lw=2, hatch='o')
                    # print(bestll_iter_num, best_loc_ll_iter_num, best_loc_ll)



                elif f == 'stdout.log' and 'my' in algorithm:
                    with open(log_file) as file:
                        num_acquisition = 0
                        for line in file:
                            if num_acquisition == 250:
                                break
                            if line.startswith('specified'):
                                spec_map[num_acquisition] = spec_map.get(num_acquisition, 0) + 1
                            else:
                                num_acquisition += 1

        if not lls:
            continue
        lls = np.array([*zip(*lls)]).T
        # print(lls)
        local_lls = np.array([*zip(*local_lls)]).T

        # if 'bayes' in algorithm:
        #     median = np.quantile(lls, 0.5, axis=0)
        #     lls -= (median[init_pts - 1] - shift)
        #     shifted = True

        starts = lls.shape[0]
        iters = list(range(lls.shape[1]))
        # if '+' in algorithm:
        #     iters = iters[39:]
        #     lls = lls[:, 39:]

        # local_iters = np.array(list(range(local_lls.shape[1]))) + lls.shape[1] - 1

        median = np.quantile(lls, 0.5, axis=0)
        # local_median = np.quantile(local_lls, 0.5, axis=0)
        # if algorithm == 'gadma':
        #     shift = median[init_pts - 1]
        if 'Bayes' in algorithm:
            mb_median = median

        # ax.plot(iters, median, label=algorithm, color=col)
        line, = ax.plot([], [], label=algorithm, color=col)
        lines.append(line)
        # ax.plot(local_iters, local_median, color=col)

        do_quar = np.quantile(lls, 0.25, axis=0)
        up_quar = np.quantile(lls, 0.75, axis=0)
        # ax.plot(iters, np.quantile(lls, 1, axis=0), label=f'best in {algorithm}', color=col)
        # ax.fill_between(iters, do_quar up_quar, alpha=0.2, color=col)  # , hatch='o')

        for_anim.append((iters, median, do_quar, up_quar))

        # local_do_quar = np.quantile(local_lls, 0.25, axis=0)
        # local_up_quar = np.quantile(local_lls, 0.75, axis=0)
        # ax.fill_between(local_iters, local_do_quar, local_up_quar, alpha=0.3, color=col, hatch='O')

        # ax.plot(iters, np.quantile(lls, 1, axis=0), label=f'best in {algorithm}', color=col)
        # plt.arrow(58, -80152.87922536346, 0, 22566.45079109667, lw=7.26, color='r', ls=':')
        # plt.arrow(58, -80152.87922536346, 0, 22566.491009141, color='r', width=1.452, head_length=800,
        #           length_includes_head=True)  # , lw=14.52)  # 1452
        # plt.arrow(193, -63101.736953612846, 0, 7789.62747188584, lw=4.90, color='g', ls=':')
        # plt.arrow(193, -63101.736953612846, 0, 7601.128710573, color='g', width=0.98, head_length=800,
        #           length_includes_head=True)  # 980

        # ax.text(58, -57586.38821622235, f'{-57586.38821622235:.2f}', color='r', fontsize=14)
        # ax.text(165, -57500.60824303997, f'{-55340.60824303997:.2f}', color='g', fontsize=14)

        min_iter_num = max(min_iter_num, cur_min_iter_num)
        min_first_ll = min(min_first_ll, min(lls[0]))

        # print(algorithm, best_model_params)
        if algorithm == 'GADMA':
            best_model_params = [0.99716247, 1.99456333, 1.49593113, 0.99716809, 0.49868243, 0.04988807, 0.09976554,
                                 0.14958237, 0.04987674]
            best_ll = -55340.60824303997
        else:
            best_model_params = [9.82948650e-01, 1.64052653e+00, 1.59686725e+00, 1.10841812e+00, 5.06491883e-01,
                                 8.76596015e-02, 4.45299012e-01, 2.02149885e-01, 5.14057289e-02]
            best_ll = -57586.38821622235

        # draw_model(data_dir, last_start_dir, algorithm, best_model_params, best_ll)

    def init():
        for line in lines:
            line.set_data([], [])
        return (*lines,)

    def anim_func(t):
        ax.collections.clear()
        for i, line in enumerate(lines):
            iters, median, do_quar, up_quar = for_anim[i]
            line.set_data(iters[:t], median[:t])

            col = f'C{color_map.get(dirnames[i], min(filter(lambda x: x + 1 not in color_map.values(), color_map.values())) + 1)}'
            ax.fill_between(iters[:t], do_quar[:t], up_quar[:t], alpha=0.2, color=col)  # , hatch='o')
        # print(x, y)
        return (*lines,)

    anim = FuncAnimation(fig, anim_func, init_func=init, frames=256, interval=2000, blit=True)
    # anim.save(os.path.join(last_start_dir, 'cool.mp4'), writer='imagemagick')

    shifted = '(bayes shifted)' if shifted else ''
    # fig.suptitle(f"{data_dir.lstrip('data_')} {starts} starts {shifted}", y=0.92)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    init_pts = 16
    init_pts = init_pts - 1

    # for k, v in spec_map.items():
    #     if v == 1 or v == 6:
    #         ax.plot(k + init_pts, mb_median[k + init_pts], '.', color='C4', mew=v, label=f'win in {v} starts')
    #     else:
    #         ax.plot(k + init_pts, mb_median[k + init_pts], '.', color='C4', mew=v)
    # min_iter_num += local_lls.shape[1]

    xts = sorted(
        set(list(
            np.array(list(filter(lambda x: x >= init_pts and x <= min_iter_num, ax.get_xticks()))[:-1]) - 1) + [0,
                                                                                                                init_pts,
                                                                                                                30 - 1,
                                                                                                                60 - 1,
                                                                                                                90 - 1,
                                                                                                                120 - 1,
                                                                                                                150 - 1,
                                                                                                                180 - 1,
                                                                                                                210 - 1,
                                                                                                                # 175 - 1,
                                                                                                                # 175 - 1,
                                                                                                                240 - 1,
                                                                                                                # 280 - 1,
                                                                                                                min_iter_num]))

    # xts = np.array([1, 50, 150, 200]) - 1
    xlabels = np.copy(xts) + 1
    # xlabels = np.copy(xts) + 1 - 16
    xlabels = map(lambda x: str(int(x)), xlabels)
    ax.set_xticks(xts)
    ax.set_xticklabels(xlabels)
    # TODO start from init_pts
    ax.set_xlim(0, min_iter_num + 1)
    # ax.set_xlim(14, min_iter_num + 1)
    # ax.set_xticks([])

    ax.set_yscale('symlog')
    ax.set_ylim(-2000000, -50000)
    # print(plt.ylim())
    min_y, max_y = ax.get_ylim()
    ymax = (np.log(abs(max_possible_ll)) - np.log(abs(min_y))) / (np.log(abs(max_y)) - np.log(abs(min_y)))

    ax.axvspan(0, init_pts, ymax=ymax,
               color='orange', linestyle='--',
               linewidth=1, label='initial design',
               alpha=0.3)

    # ax.axvline(min_iter_num - local_lls.shape[1], ymax=ymax, color='b', linestyle='--', linewidth=1, label='start local optimization')

    ax.set_ylabel(r'$\log(\mathcal{L})$', fontsize=16)
    # print(min_first_ll)
    min_ll_pow = np.floor(np.log10(int(abs(min_first_ll))))
    min_ll = np.ceil(min_first_ll / (10 ** min_ll_pow)) * (10 ** min_ll_pow)
    # print(min_ll)
    # yts = set(ax.get_yticks().tolist() + [min_ll,  max_possible_ll])
    yts = set(
        ax.get_yticks().tolist() + [min_ll, -2000000, -700000, -500000, -400000, -300000, -200000, -150000, -80000,
                                    max_possible_ll])
    # yts = set(
    # ax.get_yticks().tolist() + [min_ll, -2000000, -500000, -300000, -200000, -150000, -70000,
    #                             max_possible_ll])
    # yts = set(ax.get_yticks().tolist() + [min_ll, -600000, -400000, -300000, -200000, -150000, -70000, -50000, -40000,
    #                                       max_possible_ll])
    # yts = set(ax.get_yticks().tolist() + [-70000, -60000, -50000, -45000, -40000, -35000, -30000, -25000, -22000,
    #                                       max_possible_ll])
    # yts = set(ax.get_yticks().tolist() + [-80000, -70000, -60000, -50000, -45000, -40000, -35000, -30000, -26000,
    #                                       max_possible_ll])

    # yts = [if i == 0 or for i in enumerate(yts)]
    # yts = ax.get_yticks().tolist()
    yts = sorted(filter(lambda x: x <= max_possible_ll, yts))
    # print(yts)

    ylabels = []
    for y in yts[:-1]:
        y = int(abs(y))
        y_pow = len(str(y)) - 1  # - len(str(y).rstrip('0'))
        mult = y / (10 ** y_pow)
        if mult.is_integer():
            mult = int(mult)
        ylabels.append(f"$-10^{y_pow}$" if mult == 1 else f"$-{mult}\cdot10^{y_pow}$")

    # ylabels.append(int(max_possible_ll))
    ylabels.append(fr'$-${-max_possible_ll}')
    ax.set_yticks(yts)
    ax.set_yticklabels(ylabels)
    ax.tick_params(labelsize=18)
    ax.set_ylim(-2000000, -50000)
    # ax.yaxis.get_major_ticks()[-2].label.set_fontsize(16)
    # ax.yaxis.get_major_ticks()[-3].label.set_fontsize(16)
    ax.yaxis.get_major_ticks()[-4].label.set_fontsize(16)
    # ax.yaxis.get_major_ticks()[-5].label.set_fontsize(16)
    # ax.yaxis.get_major_ticks()[-6].label.set_fontsize(16)
    # ax.yaxis.get_major_ticks()[-7].label.set_fontsize(16)
    # ax.tick_params(axis="y", direction="in", pad=-70)
    # ax.yaxis.get_major_ticks()[-1].set_pad(-84)
    # ax.yaxis.get_major_ticks()[-6].set_pad(-50)

    ax.set_xlabel('Iteration', fontsize=16)
    # ax.xaxis.labelpad = -15

    # l = plt.legend(bbox_to_anchor=(1.001, 1), loc='upper left', borderaxespad=0.)
    # ax.legend(loc='lower right', fontsize=18)
    handles, labels = plt.gca().get_legend_handles_labels()
    # indx = np.argsort(labels)
    indx = [0, 4, 1, 2, 3]
    # indx = [0, 1]
    handles = np.array(handles)[indx]
    labels = np.array(labels)[indx]

    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower right', fontsize=18)

    # fig.savefig(os.path.join(last_start_dir, 'Iter'), bbox_inches='tight')

    # fig.savefig(os.path.join(last_start_dir, 'IterLong'), bbox_inches='tight')
    # fig.savefig(os.path.join(last_start_dir, 'bayes_comparison2'), bbox_inches='tight')
    # fig.savefig(os.path.join(last_start_dir, 'IterLoc'), bbox_inches='tight')
    # fig.savefig(os.path.join(last_start_dir, 'IterMixed'), bbox_inches='tight')

    plt.tight_layout()
    # anim.save(os.path.join(last_start_dir, 'cool.mp4'), fps=30, extra_args=['-vcodec', 'libx264'])
    anim.save(os.path.join(last_start_dir, 'cool.gif'), writer='imagemagick', fps=30, extra_args=['-vcodec', 'libx264'])

    # plt.show()


BAR_WIDTH = 0.33

width_map = {
    'gadma': -1,
    'BO+GA': 0,
    'my_bayes(MPI)': 1,
}


def draw_time(data_dir, date):
    last_start_dir, dirnames = get_dir(data_dir, date)
    dirnames = filter(lambda x: not x.startswith('_'), dirnames)
    # dirnames = sorted(dirnames)
    dirnames = ['GADMA', 'BayesOpt (MPI)', 'BayesOpt + GADMA']
    # dirnames = ['gadma']

    fig, ax = plt.subplots(figsize=(19.2, 10.8))

    max_possible_ll = int(get_max_ll(os.path.join(data_dir, 'demographic_model.py')))
    # print(max_possible_ll)

    ax.axhline(max_possible_ll, color='r', linestyle='--', linewidth=1, label=r'max possible $\log(\mathcal{L})$')

    shift = 0
    starts = 0
    init_pts = 0
    shifted = False
    init_pts = 16
    min_first_ll = 0
    init_time = 0
    max_time = 0

    for i, algorithm in enumerate(dirnames):
        col = f'C{color_map.get(algorithm, min(filter(lambda x: x + 1 not in color_map.values(), color_map.values())) + 1)}'
        color_map[algorithm] = int(col[1:])

        times = []
        lls = []
        time_ll = []
        mn = np.inf

        algo_dir = os.path.join(last_start_dir,
                                algorithm)
        for dirpath, _, files in os.walk(algo_dir):
            start_num = dirpath.split('/')[-1].lstrip('start')
            # if start_num not in map(str, [0, 1]):
            #     continue
            for f in files:
                log_file = os.path.join(dirpath, f)

                if f == 'evaluations.log':
                    with open(log_file) as file:
                        next(file)

                        cur_times = []
                        cur_lls = []
                        iter_num = 0
                        best_ll = -np.inf
                        cur_max_ll = -np.inf
                        prev_time = 0
                        need_to_add = False
                        o = False
                        for iter_num, line in enumerate(file):
                            if iter_num == 200:
                                break
                            parts = line.strip().split('\t')

                            total_time = float(parts[0])
                            if iter_num == init_pts:
                                init_time = min(init_time, total_time)
                            if need_to_add or prev_time > total_time:
                                need_to_add = True
                                # iter_change
                                # print(algorithm, 'was')
                                total_time += prev_time
                            else:
                                prev_time = total_time
                            # print(algorithm, iter_num, prev_time)

                            logLL = -float(parts[1])

                            cur_max_ll = max(cur_max_ll, logLL)
                            cur_lls.append(cur_max_ll)
                            cur_times.append(total_time)

                            if iter_num == 40:
                                mn = min(mn, total_time)

                        if iter_num > 0:
                            lls.append(cur_lls)
                            times.append(cur_times)

                            time_ll.append(list(zip(cur_times, cur_lls)))
        if not times:
            continue
        times = np.array([*zip(*times)]).T
        print(algorithm, times.shape)
        # if '+' in algorithm:
        #     print(time_ll)
        times = sorted(times.flatten())
        max_time = max(max_time, times[-1])

        lls = [[] for _ in range(len(times))]
        for i, t in enumerate(times):
            for tl in time_ll:
                le = list(filter(lambda y: y[0] <= t, tl))
                lls[i].append(le[-1][1] if le else tl[0][1])

        if '+' in algorithm:
            times = list(filter(lambda x: x > mn, times))
            lls = lls[-len(times):]

        # print(algorithm, lls)

        # starts = len(lls)
        #
        median = np.quantile(lls, 0.5, axis=1)
        # ax.bar(iters + BAR_WIDTH * width_map[algorithm], median, BAR_WIDTH, label=algorithm, color=col)
        # ax.bar(iters, median, 1, label=algorithm, color=col)
        ax.plot(times, median, label=algorithm, color=col)
        #
        do_quar = np.quantile(lls, 0.25, axis=1)
        up_quar = np.quantile(lls, 0.75, axis=1)
        # ax.plot(iters, np.quantile(lls, 1, axis=0), label=f'best in {algorithm}', color=col)
        ax.fill_between(times, do_quar, up_quar, alpha=0.2, color=col)  # , hatch='.')

        lls = np.array(lls)
        # print(lls.shape)
        # print(algorithm, np.argmin(lls), np.min(lls))
        min_first_ll = min(min_first_ll, np.min(lls[0]))

    shifted = '(bayes shifted)' if shifted else ''
    # fig.suptitle(f"{data_dir.lstrip('data_')} {starts} starts {shifted}", y=0.92)

    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # for k, v in spec_map.items():
    #     if v == 1 or v == 6:
    #     ax.plot(k + init_pts, mb_median[k + init_pts], '.', color='C4', mew=v, label=f'win in {v} starts')
    #     else:
    #         ax.plot(k + init_pts, mb_median[k + init_pts], '.', color='C4', mew=v)
    # min_iter_num += local_lls.shape[1]

    # TODO start from init_pts
    ax.set_xlim(0, 1.005 * max_time)
    # ax.set_xlim(0, 32000)
    # ax.set_xticks([])

    ax.set_yscale('symlog')
    ax.set_ylim(-1000000)

    min_y, max_y = ax.get_ylim()
    ymax = (np.log(abs(max_possible_ll)) - np.log(abs(min_y))) / (np.log(abs(max_y)) - np.log(abs(min_y)))
    # ymax = 0.960191336
    ax.axvspan(0, init_time, ymax=ymax,
               color='orange', linestyle='--',
               linewidth=1, label='initial design',
               alpha=0.3)

    min_ll_pow = np.floor(np.log10(int(abs(min_first_ll))))
    min_ll = np.ceil(min_first_ll / (10 ** min_ll_pow)) * (10 ** min_ll_pow)
    # yts = set(ax.get_yticks().tolist() + [min_ll, -90000, -80000, -70000, -60000, max_possible_ll])
    # yts = set(ax.get_yticks().tolist() + [-700000, -500000, -300000, -200000, -150000, -80000, max_possible_ll])
    yts = set(
        ax.get_yticks().tolist() + [-700000, -500000, -400000, -300000, -200000, -160000, -120000, -80000,
                                    max_possible_ll])

    yts = sorted(filter(lambda x: x <= int(max_possible_ll), yts))
    ylabels = []
    for y in yts[:-1]:
        y = int(abs(y))
        y_pow = len(str(y)) - 1  # - len(str(y).rstrip('0'))
        mult = y / (10 ** y_pow)
        if mult.is_integer():
            mult = int(mult)
        ylabels.append(f"$-10^{y_pow}$" if mult == 1 else f"$-{mult}\cdot10^{y_pow}$")

    ax.set_yticks(yts)

    ylabels.append(fr'$-${-int(max_possible_ll)}')
    ax.set_yticklabels(ylabels)
    ax.set_ylim(-1000000)

    ax.tick_params(labelsize=16)
    # ax.yaxis.set_minor_locator(MaxNLocator())

    ax.tick_params(labelsize=18)
    ax.yaxis.get_major_ticks()[-5].label.set_fontsize(16)
    ax.yaxis.get_major_ticks()[-4].label.set_fontsize(16)
    ax.set_xlabel('Total time (sec)', fontsize=16)
    # ax.xaxis.labelpad = -2

    ax.set_ylabel(r'$\log(\mathcal{L})$', fontsize=16)
    # ax.yaxis.labelpad = -2

    # ax.legend(loc='lower right', fontsize=16)
    handles, labels = plt.gca().get_legend_handles_labels()
    # indx = np.argsort(labels)
    indx = [0, 4, 2, 1, 3]
    handles = np.array(handles)[indx]
    labels = np.array(labels)[indx]

    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower right', fontsize=18)

    fig.savefig(os.path.join(last_start_dir, 'LLTimeMixed'), bbox_inches='tight')


if __name__ == '__main__':
    # data_dirs = list(filter(lambda x: x.startswith('data'), next(os.walk('.'))[1]))
    date = None

    # data_dirs = ['4_YRI_CEU_CHB_JPT_17_Jou']
    # date = '05/22/[03:15:05]'  # 4 real

    # data_dirs = ['3_YRI_CEU_CHB_13_Jou']
    # date = '05/26/[03:31:41]' # 3 real

    # data_dirs = ['data_5_DivNoMig']
    # date = '05/30/[14:54:30]'  # 5 sim 200

    data_dirs = ['data_4_DivNoMig']
    date = '04/23/[23:49:38]'
    # date = '06/03/[15:36:04]'

    # data_dirs = ['4_real_NoMig']

    # data_dirs = ['3_real_NoMig']

    # last_start_dir = os.path.join(results_dir, '05/27/[19:34:31]')

    [draw(d_d, date) for d_d in data_dirs]
    # [draw_time(d_d, date) for d_d in data_dirs]
