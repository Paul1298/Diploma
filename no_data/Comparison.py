import importlib
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

color_map = {
    'spec_not_always': 'cyan',
    'my_bayes': 'lime',
    'bayes': 'r',
    'gadma': 'g',
    'random_search': 'b'
}

def get_dir(data_dir):
    if 'results' not in os.listdir(data_dir):
        return

    results_dir = os.path.join(data_dir, 'results')
    last_mo = os.path.join(results_dir, sorted(os.listdir(results_dir))[-1])
    last_day = os.path.join(last_mo, sorted(os.listdir(last_mo))[-1])

    last_start_dir = os.path.join(last_day,
                                  sorted(os.listdir(last_day))[-1])
    # last_start_dir = os.path.join(results_dir, '04/14[12:14:15]')

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
    a = set(dirnames)
    # a.remove('random_search')
    # a.remove('gadma')
    dirnames = sorted(list(a))

    fig, ax = plt.subplots(figsize=(19.2, 10.8))
    fig.suptitle(data_dir)

    max_possible_ll = get_max_ll(os.path.join(data_dir, 'demographic_model.py'))

    plt.axhline(max_possible_ll, color='r', linestyle='--', linewidth=1, label='max_possible_ll')

    min_iter_num = np.inf

    for i, algorithm in enumerate(dirnames):
        lls = []
        iter_num = 0
        cur_min_iter_num = np.inf

        for dirpath, _, files in os.walk(os.path.join(last_start_dir,
                                                      algorithm)):
            for f in files:
                if f == 'evaluations.log':
                    log_file = os.path.join(dirpath, f)
                    with open(log_file) as file:
                        next(file)

                        cur_lls = []
                        for iter_num, line in enumerate(file):
                            parts = line.strip().split('\t')

                            logLL = -float(parts[1])
                            cur_max_ll = max(logLL, cur_lls[-1]) if cur_lls else logLL
                            cur_lls.append(cur_max_ll)

                        if iter_num != 0:
                            lls.append(cur_lls)
                            cur_min_iter_num = min(cur_min_iter_num, iter_num + 1)
                elif f == 'params.py':
                    params_file = os.path.join(dirpath, f)
                    init_pts = get_params(params_file)

        if not lls:
            continue
        lls = np.array([*zip(*lls)]).T

        ll_mean = lls.mean(axis=0)

        iters = list(range(lls.shape[1]))

        col = color_map[algorithm]

        plt.plot(iters, ll_mean, label=algorithm)

        # confidence interval
        ci = 1.96 * np.std(lls, axis=0) / np.sqrt(lls.shape[0])
        if np.any(ll_mean + ci > max_possible_ll):
            ci = 1.96 * np.std(lls, axis=0) / lls.shape[0]

        plt.fill_between(iters, (ll_mean - ci), (ll_mean + ci), alpha=0.2)

        min_iter_num = min(min_iter_num, cur_min_iter_num)

    plt.yscale('symlog')

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    init_pts = init_pts - 1
    plt.xticks(sorted(list(filter(lambda x: x >= init_pts and x <= min_iter_num, plt.xticks()[0]))[:-1] + [0, init_pts, min_iter_num]))
    plt.xlim(xmax=min_iter_num + 1)

    plt.yticks(list(filter(lambda x: x < max_possible_ll, plt.yticks()[0])) + [int(max_possible_ll)])

    min_y, max_y = plt.ylim()
    plt.axvspan(0, init_pts, ymax=((max_possible_ll - min_y) / (max_y - min_y)),
                color='orange', linestyle='--',
                linewidth=1, label='init_pts',
                alpha=0.3)

    plt.xlabel('Iteration')
    plt.ylabel('LogLL')

    # l = plt.legend(bbox_to_anchor=(1.001, 1), loc='upper left', borderaxespad=0.)
    l = plt.legend(loc='best')

    # textstr = '\n'.join((
    #     r'$\mu=%.2f$' % (mu,),
    #     r'$\mathrm{median}=%.2f$' % (median,),
    #     r'$\sigma=%.2f$' % (sigma,)))

    plt.savefig(os.path.join(last_start_dir, 'Iteration2'), bbox_inches='tight')


def draw_iter(d_d):
    pass


BAR_WIDTH = 0.25

width_map = {
    'my_bayes': -2 * BAR_WIDTH,
    'bayes': -BAR_WIDTH,
    'gadma': 0,
    'random_search': BAR_WIDTH
}

# def mean_iter_time(data_dir):
#     last_start_dir, dirnames = get_dir(data_dir)
#
#     fig, ax = plt.subplots(figsize=(19.2, 10.8))
#     fig.suptitle(data_dir)
#
#     plot_lines = []
#     label_names = []
#
#     max_iter_num = 0
#
#     for algorithm in dirnames:
#         iter_times = []
#         iter_num = 0
#         cur_max_iter_num = 0
#         avg_iter_num = (0, 0)
#
#         for dirpath, _, files in os.walk(os.path.join(last_start_dir,
#                                                       algorithm)):
#             for f in files:
#                 if f == 'evaluations.log':
#                     log_file = os.path.join(dirpath, f)
#                     with open(log_file) as file:
#                         next(file)
#
#                         cur_iter_times = []
#                         for iter_num, line in enumerate(file):
#                             parts = line.strip().split('\t')
#
#                             iter_time = float(parts[-1])
#                             cur_iter_times.append(iter_time)
#
#                         iter_times.append(cur_iter_times)
#                         cur_max_iter_num = max(cur_max_iter_num, iter_num + 1)
#                         avg_iter_num = (avg_iter_num[0] + iter_num + 1, avg_iter_num[1] + 1)
#
#         # lls = np.array([*itertools.zip_longest(*lls, fillvalue=max_possible_ll)]).T
#         iter_times = np.array([*zip(*iter_times)]).T
#
#         iter_time_mean = iter_times.mean(axis=0)
#
#         iters = np.array(range(iter_times.shape[1]))
#
#         col = color_map[algorithm]
#         width = width_map[algorithm]
#
#         plot_lines.append(plt.bar(iters + width, iter_time_mean, color=col, width=BAR_WIDTH)[0])
#         label_names.append(f'{algorithm}: {avg_iter_num[0] / avg_iter_num[1]} avg iters')
#
#         # confidence interval
#         ci = 1.96 * np.std(iter_times, axis=0) / np.sqrt(iter_times.shape[0])
#
#         max_iter_num = max(max_iter_num, cur_max_iter_num)
#
#     iters = list(range(max_iter_num))
#     plt.xticks(iters)
#
#     plt.xlabel('Iteration')
#     plt.ylabel('Iteration time, seconds')
#
#     plt.legend(plot_lines, label_names)
#
#     plt.savefig(os.path.join(last_start_dir, 'IterTime'), bbox_inches='tight')


# def time(data_dir):
#     last_start_dir, dirnames = get_dir(data_dir)
#
#     fig, ax = plt.subplots(figsize=(19.2, 10.8))
#     fig.suptitle(data_dir)
#
#     max_possible_ll = get_max_ll(os.path.join(data_dir, 'demographic_model.py'))
#
#     max_line = plt.axhline(max_possible_ll, color='r', linestyle='--', linewidth=1)
#     plot_lines = [max_line]
#     label_names = ['max_possible_ll']
#
#     max_iter_num = 0
#
#     for algo_dir in dirnames:
#         lls = []
#         times = []
#
#         iter_num = 0
#         cur_max_iter_num = 0
#         avg_iter_num = (0, 0)
#
#         for dirpath, _, files in os.walk(os.path.join(last_start_dir,
#                                                       algo_dir)):
#             for f in files:
#                 if f == 'evaluations.log':
#                     log_file = os.path.join(dirpath, f)
#                     with open(log_file) as file:
#                         next(file)
#
#                         cur_lls = []
#                         cur_time = []
#                         for iter_num, line in enumerate(file):
#                             parts = line.strip().split('\t')
#
#                             total_time = float(parts[0])
#                             cur_time.append(total_time)
#
#                             logLL = -float(parts[1])
#                             cur_max_ll = max(logLL, cur_lls[-1]) if cur_lls else logLL
#                             cur_lls.append(cur_max_ll)
#
#                         times.append(cur_time)
#
#                         lls.append(cur_lls)
#                         cur_max_iter_num = max(cur_max_iter_num, iter_num + 1)
#                         avg_iter_num = (avg_iter_num[0] + iter_num + 1, avg_iter_num[1] + 1)
#
#         col = color_map[algo_dir]
#         [plt.plot(*z, color=col) for z in zip(times, lls)]
#
#     plt.yscale('symlog')
#
#     plt.xlabel('Seconds')
#     plt.ylabel('LogLL')
#
#     plt.savefig(os.path.join(last_start_dir, 'Time'))

if __name__ == '__main__':
    # data_dirs = list(filter(lambda x: x.startswith('data'), next(os.walk('.'))[1]))
    data_dirs = ['data_4_DivNoMig']
    # data_dirs = ['data_2_DivMigr']

    [draw(d_d) for d_d in data_dirs]
    # [mean_iter_time(d_d) for d_d in data_dirs]
    # [time(d_d) for d_d in data_dirs]