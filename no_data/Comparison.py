import importlib
import os

import matplotlib.pyplot as plt
import moments
import numpy as np
from matplotlib.ticker import MaxNLocator

# {'C0',   'C1',     'C2',    'C3',  'C4',     'C5',    'C6',   'C7',   'C8',    'C9'}
# {'blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'}
color_map = {
    'my_bayes(MPI)': 3,
    'my_bayes(EI)': 5,
    'BO+GA': 9,

    'gadma': 2,
    'random_search': 0
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


def draw_model(data_dir, algo_dir, best_model_params):
    model_file = os.path.join(data_dir, 'demographic_model.py')

    dem_model = importlib.import_module(model_file.replace(os.path.sep, '.').rstrip('.py'))

    N_A = 11293

    # Draw model
    samples = 20
    model = moments.ModelPlot.generate_model(dem_model.model_func, best_model_params,
                                             (samples, samples, samples, samples))

    save_file = f'{algo_dir}_model.png'
    fig, ax = plt.subplots()
    moments.ModelPlot.plot_model(model,
                                 ax=ax,
                                 # save_file=save_file,
                                 fig_title='',
                                 pop_labels=['YRI', 'CEU', 'CHB', 'JPT'],
                                 nref=N_A,
                                 draw_scale=True,
                                 gen_time=0.029,
                                 gen_time_units="Thousand years",
                                 grid=False,
                                 reverse_timeline=True)
    fig.savefig(save_file, bbox_inches='tight')


def draw(data_dir, date):
    last_start_dir, dirnames = get_dir(data_dir, date)
    dirnames = filter(lambda x: not x.startswith('_'), dirnames)
    # a.remove('random_search')
    # a.remove('gadma')
    dirnames = sorted(dirnames)

    fig, ax = plt.subplots(figsize=(19.2, 10.8))

    max_possible_ll = get_max_ll(os.path.join(data_dir, 'demographic_model.py'))

    # ax.axhline(max_possible_ll, color='r', linestyle='--', linewidth=1, label='max_possible_ll')

    min_iter_num = 0
    shift = 0
    starts = 0
    init_pts = 0
    shifted = False
    min_first_ll = 0

    spec_map = {}
    for i, algorithm in enumerate(sorted(dirnames)):
        # if algorithm == 'my_bayes(EI)':
        #    continue
        lls = []
        iter_num = 0
        cur_min_iter_num = np.inf
        q = 0
        algo_dir = os.path.join(last_start_dir,
                                algorithm)
        best_ll = -np.inf
        for dirpath, _, files in os.walk(algo_dir):
            # end = dirpath.split('/')[-1][5:]
            # # print(end, 2)
            # if end and int(end) >= 4:
            #    continue
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
                        for iter_num, line in enumerate(file):
                            if iter_num == 66:
                                break
                            parts = line.strip().split('\t')

                            logLL = -float(parts[1])

                            cur_max_ll = max(cur_max_ll, logLL) if cur_lls else logLL
                            cur_lls.append(cur_max_ll)

                            if logLL > best_ll:
                                best_ll = logLL
                                best_model_params = np.fromstring(parts[2][1:-1], dtype=float, sep=',')

                        if iter_num > 0:
                            lls.append(cur_lls)
                            cur_min_iter_num = min(cur_min_iter_num, iter_num - 1)
                elif f == 'params.py':
                    init_pts = get_params(log_file)

                elif f == 'stdout.log' and 'my' in algorithm:
                    with open(log_file) as file:
                        num_acquisition = 0
                        for line in file:
                            if num_acquisition == 50:
                                break
                            if line.startswith('specified'):
                                spec_map[num_acquisition] = spec_map.get(num_acquisition, 0) + 1
                            else:
                                num_acquisition += 1

        if not lls:
            continue
        lls = np.array([*zip(*lls)]).T

        # if 'bayes' in algorithm:
        #     median = np.quantile(lls, 0.5, axis=0)
        #     lls -= (median[init_pts - 1] - shift)
        #     shifted = True

        starts = lls.shape[0]
        iters = list(range(lls.shape[1]))

        col = f'C{color_map.get(algorithm, min(filter(lambda x: x + 1 not in color_map.values(), color_map.values())) + 1)}'
        color_map[algorithm] = int(col[1:])

        median = np.quantile(lls, 0.5, axis=0)
        # if algorithm == 'gadma':
        #     shift = median[init_pts - 1]
        if 'my' in algorithm:
            mb_median = median

        ax.plot(iters, median, label=algorithm, color=col)

        do_quar = np.quantile(lls, 0.25, axis=0)
        up_quar = np.quantile(lls, 0.75, axis=0)
        ax.fill_between(iters, do_quar, up_quar, alpha=0.2, color=col)

        min_iter_num = max(min_iter_num, cur_min_iter_num)
        min_first_ll = min(min_first_ll, min(lls[0]))

        print(algorithm, best_model_params)
        # draw_model(data_dir, algo_dir, best_model_params)

    shifted = '(bayes shifted)' if shifted else ''
    # fig.suptitle(f"{data_dir.lstrip('data_')} {starts} starts {shifted}", y=0.92)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    init_pts = 16
    init_pts = init_pts - 1

    for k, v in spec_map.items():
        if v == 1 or v == 6:
            ax.plot(k + init_pts, mb_median[k + init_pts], '.', color='C4', mew=v, label=f'win in {v} starts')
        else:
            ax.plot(k + init_pts, mb_median[k + init_pts], '.', color='C4', mew=v)

    xts = sorted(
        set(list(
            np.array(list(filter(lambda x: x >= init_pts and x <= min_iter_num, ax.get_xticks()))[:-1]) - 1) + [0,
                                                                                                                init_pts,
                                                                                                                min_iter_num]))
    # xlabels = np.copy(xts) + 1
    xlabels = np.copy(xts) + 1 - 16
    xlabels = map(lambda x: str(int(x)), xlabels)
    ax.set_xticks(xts)
    ax.set_xticklabels(xlabels)
    # TODO start from init_pts
    # ax.set_xlim(0, min_iter_num + 1)
    ax.set_xlim(14, min_iter_num + 1)
    # ax.set_xticks([])

    # ax.set_yscale('symlog')
    # print(plt.ylim())
    min_y, max_y = ax.get_ylim()
    ymax = (np.log(abs(max_possible_ll)) - np.log(abs(min_y))) / (np.log(abs(max_y)) - np.log(abs(min_y)))

    # ax.axvspan(0, init_pts, ymax=ymax,
    #            color='orange', linestyle='--',
    #            linewidth=1, label='init_pts',
    #            alpha=0.3)
    ax.set_ylabel(r'$\log(\mathcal{L})$', fontsize=16)
    # print(min_first_ll)
    min_ll_pow = np.floor(np.log10(int(abs(min_first_ll))))
    min_ll = np.ceil(min_first_ll / (10 ** min_ll_pow)) * (10 ** min_ll_pow)
    # print(min_ll)
    yts = ax.get_yticks().tolist() + [-500000, -200000, -80000, int(max_possible_ll)]
    # yts = ax.get_yticks().tolist() + [-500000, -200000, -80000, int(max_possible_ll)]
    # yts = ax.get_yticks().tolist() + [min_ll, -500000, -200000, int(max_possible_ll)]
    # yts = ax.get_yticks().tolist() + [min_ll, -60000, -40000, -30000, int(max_possible_ll)]

    # yts = [if i == 0 or for i in enumerate(yts)]
    yts = sorted(filter(lambda x: x <= int(max_possible_ll), yts))
    # print(yts)

    ax.set_ylim(-520000, -100000)
    yts = ax.get_yticks().tolist()
    ylabels = []
    for y in yts:
        y = int(abs(y))
        y_pow = len(str(y)) - 1
        mult = int(str(y).rstrip('0'))
        mult = mult if len(str(mult)) == 1 else float(mult) / 10
        ylabels.append(f"$-10^{y_pow}$" if mult == 1 else f"$-{mult}\cdot10^{y_pow}$")

    # ylabels.append(int(max_possible_ll))
    ax.set_yticks(yts)
    ax.set_yticklabels(ylabels)
    plt.tick_params(labelsize=18)

    ax.set_xlabel('Optimization iteration', fontsize=16)

    # l = plt.legend(bbox_to_anchor=(1.001, 1), loc='upper left', borderaxespad=0.)
    # ax.legend(loc='lower right', fontsize=14)
    handles, labels = plt.gca().get_legend_handles_labels()
    indx = np.argsort(labels)
    handles = np.array(handles)[indx]
    labels = sorted(labels)

    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower right', fontsize=18)

    # fig.savefig(os.path.join(last_start_dir, 'IterLong'), bbox_inches='tight')
    fig.savefig(os.path.join(last_start_dir, 'IterOnlyBayes4'), bbox_inches='tight')


if __name__ == '__main__':
    # data_dirs = list(filter(lambda x: x.startswith('data'), next(os.walk('.'))[1]))
    date = None

    # data_dirs = ['4_YRI_CEU_CHB_JPT_17_Jou']
    # date = '05/22/[03:15:05]'  # 4 real

    # data_dirs = ['3_YRI_CEU_CHB_13_Jou']
    # last_start_dir = os.path.join(results_dir, '05/26/[03:31:41]') # 3 real

    # data_dirs = ['data_5_DivNoMig']
    # date = '05/30/[14:54:30]'  # 5 sim 200

    data_dirs = ['data_4_DivNoMig']
    date = '04/23/[23:49:38]'

    # data_dirs = ['4_real_NoMig']

    # data_dirs = ['3_real_NoMig']

    # last_start_dir = os.path.join(results_dir, '05/27/[19:34:31]')

    [draw(d_d, date) for d_d in data_dirs]
