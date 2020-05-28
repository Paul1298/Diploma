import importlib
import multiprocessing as mp
import os
import sys
import time
from functools import partial
from shutil import copy

import gadma
import moments
import numpy as np

import Inference
from DemModelUpdater import Updater

SKIP_DIRS = ['__pycache__']

KNOWN_ALGORITHMS = {
    'my_bayes(EI)': lambda *args, **kwargs: KNOWN_ALGORITHMS['my_bayes(MPI)'](*args, **kwargs, acqu_type='EI'),
    'bayes(EI)': lambda *args, **kwargs: KNOWN_ALGORITHMS['bayes'](*args, **kwargs, acqu_type='EI'),
    'spec_not_always': lambda *args, **kwargs: KNOWN_ALGORITHMS['my_bayes(MPI)'](*args, **kwargs,
                                                                            spec_anchors_always=False),

    'my_bayes(MPI)': lambda *args, **kwargs: KNOWN_ALGORITHMS['bayes'](*args, **kwargs, my=True),

    'bayes': lambda data, func, lower_bound, upper_bound, p_ids, p0, ll0, iter_num, num_init_pts, filename, **kwargs:
    Inference.optimize_bayes(data, func,
                             lower_bound, upper_bound, p_ids,
                             max_iter=iter_num,
                             p0=p0,
                             ll0=ll0,
                             num_init_pts=num_init_pts,
                             output_log_file=filename,
                             **kwargs),

    'gadma': lambda data, func, lower_bound, upper_bound, p_ids, p0, ll0, iter_num, num_init_pts, filename:
    gadma.Inference.optimize_ga(len(p_ids), data, func,
                                lower_bound=lower_bound,
                                upper_bound=upper_bound,
                                X_init=p0,
                                Y_init=ll0,

                                p_ids=p_ids,
                                optimization_name=None,
                                maxeval=iter_num,
                                num_init_pts=num_init_pts,
                                output_log_file=filename),

    'random_search': lambda data, func, lower_bound, upper_bound, p_ids, p0, ll0, iter_num, num_init_pts, filename:
    gadma.Inference.optimize_ga(len(p_ids), data, func,
                                lower_bound=lower_bound,
                                upper_bound=upper_bound,
                                X_init=p0,
                                Y_init=ll0,
                                p_ids=p_ids,

                                frac_of_old_models=0,
                                frac_of_mutated_models=0,
                                frac_of_crossed_models=0,
                                # Единица минус все эти доли равно доле случайных и обнулив их
                                # мы делаем 100% случайных моделей на каждом шаге, а это и есть случайный поиск
                                optimization_name=None,
                                maxeval=iter_num,
                                num_init_pts=num_init_pts,
                                output_log_file=filename)
}


def parallel_wrap(fun, args):
    return fun(*args)


def log_print(bench_log_path, context):
    with open(bench_log_path, 'a') as log_file:
        print(time.strftime('[%X]') + context, file=log_file)


def generate_p0(data_dir, num_init_pts):
    for dirpath, dirs, files in os.walk(data_dir):
        if any(x in dirpath for x in SKIP_DIRS + ['results']):
            dirs[:] = []
            continue
        data_fs_file, model_file, _ = map(lambda x: os.path.join(dirpath, x), sorted(files))
        dem_model = importlib.import_module(model_file.replace(os.path.sep, '.').rstrip('.py'))

        p0 = np.array(
            [[Inference.generate_random_value(low_bound, upp_bound, p_id) for (low_bound, upp_bound, p_id)
              in zip(dem_model.lower_bound, dem_model.upper_bound, dem_model.p_ids)]
             for _ in range(num_init_pts)])

        data = moments.Spectrum.from_file(data_fs_file)
        ns = data.sample_sizes
        log = True

        # TODO with open(self.log_file, 'w') as log_file:
        obj_func = partial(Inference.objective_func, dem_model.model_func, data, ns, log)

        ll0 = obj_func(p0)
        return p0, ll0


def run(data_dir, algorithm=None, start_idx=0, start_time=None, p0=None, ll0=None, iter_num=50, num_init_pts=16,
        output_log_dir='results'):
    for dirpath, dirs, files in os.walk(data_dir):
        if any(x in dirpath for x in SKIP_DIRS + [output_log_dir]):
            dirs[:] = []
            continue

        # TODO: do it only when is output_log_dir a subdirectory of data_dir
        result_dir = os.path.join(dirpath, output_log_dir, start_time)

        bench_log_path = os.path.join(result_dir, 'bench.log')
        cur_log_print = partial(log_print, bench_log_path)
        """
        Предполагаем, что данные хранятся так:

        data_dir_name
            -data.fs
            -demographic_model.py
            -simulate_with_moments.py
            -...
        """
        try:
            data_fs_file, model_file = map(lambda x: os.path.join(dirpath, x), ['data.fs', 'demographic_model.py'])
        except ValueError:
            cur_log_print('Unsupported structure')
            continue

        dem_model = importlib.import_module(model_file.replace(os.path.sep, '.').rstrip('.py'))

        # Load the no_data
        data = moments.Spectrum.from_file(data_fs_file)

        cur_log_print(f'Start {start_idx} for {algorithm} algorithm')

        optim = KNOWN_ALGORITHMS.get(algorithm)
        if optim is None:
            cur_log_print("Unknown algorithm")
        else:
            algorithm_dir = os.path.join(result_dir, algorithm)
            log_dir = os.path.join(algorithm_dir, f'start{start_idx}')
            os.makedirs(log_dir, exist_ok=True)
            std_path = os.path.join(log_dir, 'stdout.log')
            sys.stdout = open(std_path, 'w')

            t1 = time.time()
            opt = optim(data, dem_model.model_func,
                        dem_model.lower_bound, dem_model.upper_bound, dem_model.p_ids,
                        p0, ll0, iter_num, num_init_pts,
                        os.path.join(log_dir, 'evaluations.log'))
            if 'bayes' in algorithm:
                opt.plot_convergence(os.path.join(log_dir, 'convergence.png'))
            t2 = time.time()
            cur_log_print(f'\tEvaluation time: {t2 - t1} ({start_idx} for {algorithm})')
        cur_log_print(f'Finish {start_idx} for {algorithm} algorithm')


def pre_run(data_dir, start_time, output_log_dir='results'):
    for dirpath, dirs, files in os.walk(data_dir):
        if any(x in dirpath for x in SKIP_DIRS + [output_log_dir]):
            dirs[:] = []
            continue

        model_file, sim_file = map(lambda x: os.path.join(dirpath, x),
                                   ['demographic_model.py', 'simulate_with_moments.py'])

        # save version of code, which it was launched
        result_dir = os.path.join(dirpath, output_log_dir, start_time)
        os.makedirs(result_dir, exist_ok=True)
        # TODO разные версии Benchmark
        copy('Benchmark.py', os.path.join(result_dir, 'CurrentVersionBenchmark.py'))

        sys.path.insert(0, dirpath)
        Updater(model_file).check_model(sim_file)


# def update_data_structure(data_dir, output_log_dir='results'):
#     result_dir = os.path.join(data_dir, output_log_dir)
#     for dirpath, dirs, files in os.walk(result_dir):
#         if dirpath[-1] == ']':
#             last_dir = dirpath.split('/')[-1]
#             mo = last_dir[:2]
#             day = last_dir[3:5]
#             rest = last_dir[5:]
#             dst = os.path.join(result_dir, mo, day, rest)
#             os.makedirs(dst, exist_ok=True)
#             for f in os.listdir(dirpath):
#                 move(os.path.join(dirpath, f), dst)

def get_init(data_dir, start_time, num_init_pts, output_log_dir='results', base_algo='gadma'):
    algo_dir = os.path.join(data_dir, output_log_dir, start_time, base_algo)

    res = {}
    for dirpath, dirnames, files in os.walk(algo_dir):
        last_dir = dirpath.split('/')[-1]
        start_word = 'start'
        if last_dir.startswith(start_word):
            cur_start = int(last_dir[len(start_word):])
            p0, logLL = [], []
            for f in files:
                if f == 'evaluations.log':
                    log_file = os.path.join(dirpath, f)
                    with open(log_file) as file:
                        next(file)

                        for iter_num, line in enumerate(file):
                            parts = line.strip().split('\t')

                            logLL.append(float(parts[1]))
                            p0.append(np.fromstring(parts[2][1:-1], dtype=float, sep=','))
                            if iter_num == num_init_pts - 1:
                                break
            res[(data_dir, cur_start)] = (np.array(p0), np.array(logLL))
    return res


def get_calced(data_dir, start_time, algo, output_log_dir='results'):
    algo_dir = os.path.join(data_dir, output_log_dir, start_time, algo)
    res = []
    for dirpath, dirnames, files in os.walk(algo_dir):
        last_dir = dirpath.split('/')[-1]
        start_word = 'start'
        if last_dir.startswith(start_word):
            cur_start = int(last_dir[len(start_word):])
            res.append(cur_start)
    return res


if __name__ == '__main__':
    # TODO replace with os delim
    # start_time = time.strftime('%m/%d/[%X]')
    # start_time = '04/23/[23:49:38](BOGA)'
    start_time = '05/28/[01:57:54]'
    # data_dirs = list(filter(lambda x: x.startswith('data'), next(os.walk('.'))[1]))
    # data_dirs = ['data_4_DivNoMig']
    data_dirs = ['4_real_NoMig']
    # data_dirs = ['4_YRI_CEU_CHB_JPT_17_Jou']
    [pre_run(d, start_time) for d in data_dirs]

    algos = ['random_search']
    # algos = ['my_bayes(MPI)', 'my_bayes(EI)', 'bayes', 'bayes(EI)']
    # algos = ['gadma', 'random_search']

    num_starts = 16

    iter_num = 200
    num_init_pts = 10

    # res = get_init(data_dirs[0], '04/23/[23:49:38]', 40, base_algo='my_bayes(MPI)')
    # res = get_init(data_dirs[0], '05/28/[01:57:54]', 10, base_algo='gadma')

    # starts_arr = range(num_starts)
    # calced = get_calced(data_dirs[0], start_time, algos[0])
    # starts_arr = list(set(x[1] for x in res.keys()) - set(calced))[:num_starts]
    starts_arr = list(range(4, 20))
    print(start_time, data_dirs, algos, starts_arr)

    num_processes = 8
    pool = mp.Pool(num_processes)

    # res = pool.map(partial(parallel_wrap, generate_p0),
    #                [(d, small_num_init_pts) for d in data_dirs for i in range(num_starts)])
    # pairs = [(d, i) for d in data_dirs for i in range(num_starts)]
    # p0s = dict(zip(pairs, [x[0] for x in res]))
    # ll0s = dict(zip(pairs, [x[1] for x in res]))

    # X = [(d, a, i, start_time, p0s[(d, i)], ll0s[(d, i)]) for d in data_dirs for a in algos for i in range(num_starts)]

    # # print(starts_arr)
    # X = [(d, a, i, start_time, res[(d, i)][0], res[(d, i)][1]) for d in data_dirs for a in algos for i in starts_arr]
    X = [(d, a, i, start_time) for d in data_dirs for a in algos for i in starts_arr]
    kwargs = {'iter_num': iter_num, 'num_init_pts': num_init_pts}

    pool.map(partial(parallel_wrap, partial(run, **kwargs)), X)
    pool.close()
    pool.join()
