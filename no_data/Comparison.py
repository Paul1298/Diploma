import importlib
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# {'C0',   'C1',     'C2',    'C3',  'C4',     'C5',    'C6',   'C7',   'C8',    'C9'}
# {'blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'}
color_map = {
    'my_bayes(MPI)': 3,
    # 'my_bayes(EI)': 5,
    'BO+GA': 9,

    'gadma': 2,
    'random_search': 0
}


def get_dir(data_dir):
    if 'results' not in os.listdir(data_dir):
        return

    results_dir = os.path.join(data_dir, 'results')
    last_mo = os.path.join(results_dir, sorted(os.listdir(results_dir))[-1])
    last_day = os.path.join(last_mo, sorted(os.listdir(last_mo))[-1])

    last_start_dir = os.path.join(last_day,
                                  sorted(os.listdir(last_day))[-1])
    last_start_dir = os.path.join(results_dir, '04/23/[23:49:38]')
    # last_start_dir = os.path.join(results_dir, '05/27/[19:34:31]')

    dirnames = next(os.walk(last_start_dir))[1]
    return last_start_dir, dirnames


def get_max_ll(model_file):
    dem_model = importlib.import_module(model_file.replace('/', '.').rstrip('.py'))
    return dem_model.__getattribute__('max_possible_ll')


def get_params(params_file):
    name = params_file.replace('/', '.').rstrip('.py')
    dem_model = importlib.import_module(name)
    return dem_model.__getattribute__('num_init_pts')


def draw(data_dir):
    last_start_dir, dirnames = get_dir(data_dir)
    a = filter(lambda x: not x.startswith('_'), dirnames)
    # a.remove('random_search')
    # a.remove('gadma')
    dirnames = sorted(a)

    fig, ax = plt.subplots(figsize=(19.2, 10.8))

    max_possible_ll = get_max_ll(os.path.join(data_dir, 'demographic_model.py'))

    plt.axhline(max_possible_ll, color='r', linestyle='--', linewidth=1, label='max_possible_ll')

    min_iter_num = 0
    shift = 0
    starts = 0
    init_pts = 0
    shifted = False
    for i, algorithm in enumerate(sorted(dirnames)):
        # if algorithm == 'my_bayes(EI)':
        #    continue
        lls = []
        iter_num = 0
        cur_min_iter_num = np.inf
        q = 0
        for dirpath, _, files in os.walk(os.path.join(last_start_dir,
                                                      algorithm)):
            # end = dirpath.split('/')[-1][5:]
            # # print(end, 2)
            # if end and int(end) >= 4:
            #    continue
            for f in files:
                if f == 'evaluations.log':
                    q += 1
                    # if q >= 4:
                    #     break
                    log_file = os.path.join(dirpath, f)
                    with open(log_file) as file:
                        next(file)

                        cur_lls = []
                        iter_num = 0
                        for iter_num, line in enumerate(file):
                            if iter_num + 1 == 150:
                                break
                            parts = line.strip().split('\t')

                            logLL = -float(parts[1])
                            cur_max_ll = max(logLL, cur_lls[-1]) if cur_lls else logLL
                            cur_lls.append(cur_max_ll)

                        if iter_num > 0:
                            lls.append(cur_lls)
                            cur_min_iter_num = min(cur_min_iter_num, iter_num)
                elif f == 'params.py':
                    params_file = os.path.join(dirpath, f)
                    init_pts = get_params(params_file)

        if not lls:
            continue
        lls = np.array([*zip(*lls)]).T

        # if 'bayes' in algorithm:
        #     median = np.quantile(lls, 0.5, axis=0)
        #     lls -= (median[init_pts - 1] - shift)
        #     shifted = True

        starts = lls.shape[0]
        iters = list(range(lls.shape[1]))

        col = f'C{color_map.get(algorithm, max(color_map.values()) + 1)}'
        color_map[algorithm] = int(col[1:])

        median = np.quantile(lls, 0.5, axis=0)
        # if algorithm == 'gadma':
        #     shift = median[init_pts - 1]

        plt.plot(iters, median, label=algorithm, color=col)

        do_quar = np.quantile(lls, 0.25, axis=0)
        up_quar = np.quantile(lls, 0.75, axis=0)
        plt.fill_between(iters, do_quar, up_quar, alpha=0.2, color=col)

        min_iter_num = max(min_iter_num, cur_min_iter_num)

    shifted = '(bayes shifted)' if shifted else ''
    # fig.suptitle(f"{data_dir.lstrip('data_')} {starts} starts {shifted}", y=0.92)
    plt.yscale('symlog')

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    init_pts = 16
    init_pts = init_pts - 1
    xts = sorted(set(list(filter(lambda x: x >= init_pts and x <= min_iter_num, plt.xticks()[0]))[:-1] + [0, init_pts,
                                                                                                          min_iter_num]))
    xlabels = np.copy(xts) + 1
    xlabels = map(lambda x: str(int(x)), xlabels)
    plt.xticks(xts, xlabels)
    # TODO start from init_pts
    plt.xlim(0, min_iter_num + 1)

    min_y, max_y = plt.ylim()
    ymax = (np.log(abs(max_possible_ll)) - np.log(abs(min_y))) / (np.log(abs(max_y)) - np.log(abs(min_y)))

    plt.axvspan(0, init_pts, ymax=ymax,
                color='orange', linestyle='--',
                linewidth=1, label='init_pts',
                alpha=0.3)
    plt.ylabel('LogLL')
    yts = plt.yticks()[0].tolist() + [int(max_possible_ll)]
    yts = sorted(filter(lambda x: x <= int(max_possible_ll), yts))

    ylabels = [item.get_text() for item in plt.yticks()[1]]
    ylabels[-1] = int(max_possible_ll)
    ylabels[:-1] = list(map(lambda x: f"$-10^{{{int(np.log10(abs(x)))}}}$", yts[:-1]))
    plt.yticks(yts, ylabels)

    plt.xlabel('Iteration')

    # l = plt.legend(bbox_to_anchor=(1.001, 1), loc='upper left', borderaxespad=0.)
    l = plt.legend(loc='best')

    plt.savefig(os.path.join(last_start_dir, 'Iter'), bbox_inches='tight')


if __name__ == '__main__':
    # data_dirs = list(filter(lambda x: x.startswith('data'), next(os.walk('.'))[1]))
    # data_dirs = ['4_YRI_CEU_CHB_JPT_17_Jou']
    # data_dirs = ['3_YRI_CEU_CHB_13_Jou']
    data_dirs = ['data_4_DivNoMig']
    # data_dirs = ['4_real_NoMig']

    [draw(d_d) for d_d in data_dirs]
