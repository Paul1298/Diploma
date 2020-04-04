import importlib
import os

import matplotlib.pyplot as plt

color_map = {
    'bayes': 'r',
    'gadma': 'g',
    'random_search': 'b',
}


def get_max_ll(model_file):
    dem_model = importlib.import_module(model_file.replace('/', '.').rstrip('.py'))
    return dem_model.__getattribute__('max_ll')


def draw(data_dir):
    results_dir = os.path.join(data_dir, 'results')
    last_start_dir = os.path.join(results_dir,
                                  sorted(os.listdir(results_dir))[-1])

    fig, ax = plt.subplots(figsize=(21, 12))
    fig.suptitle('Comparison')

    max_ll = get_max_ll(os.path.join(data_dir, 'demographic_model.py'))
    plt.axhline(max_ll, color='r', linestyle='--', label='max_ll')

    color_num = 0
    max_iter_num = 0

    dirnames = next(os.walk(last_start_dir))[1]

    for algo_dir in dirnames:
        lls = []

        for dirpath, _, files in os.walk(os.path.join(last_start_dir,
                                                      algo_dir)):
            for f in files:
                if f == 'evaluations.log':
                    log_file = os.path.join(dirpath, f)

                    iter_num = 0
                    cur_lls = []

                    with open(log_file) as file:
                        next(file)

                        for line in file:
                            parts = line.strip().split('\t')

                            logLL = float(parts[1])
                            min_ll = min(logLL,
                                         cur_lls[-1]) if cur_lls else logLL
                            cur_lls.append(min_ll)

                            iter_num += 1

                    lls.append(cur_lls)

        lls = -np.array(lls)

        ll_min = lls.min(axis=0)
        ll_max = lls.max(axis=0)
        ll_mean = lls.mean(axis=0)

        # TODO: repeat(or special values) last iters if iter_num is different
        max_iter_num = max(max_iter_num, iter_num)

        iters = list(range(iter_num))
        col = color_map[algo_dir]
        #         print()

        plt.plot(iters, ll_mean, label=algo_dir, color=col)

        plt.fill_between(iters, ll_min, ll_max, alpha=0.4, color=col)
        color_num += 1

    #     ax.yaxis.set_major_formatter(major_formatter)
    #     plt.yscale('log')

    iters = list(range(max_iter_num))
    plt.xticks(iters)

    plt.xlabel('Iteration')
    plt.ylabel('LogLL')

    plt.legend()

    plt.savefig(os.path.join(last_start_dir, 'Comparison'))


if __name__ == '__main__':
    data_dirs = list(filter(lambda x: x.startswith('data'), next(os.walk('.'))[1]))
    [draw(d_d) for d_d in data_dirs]
