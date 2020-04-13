import importlib
import multiprocessing as mp
import os
import sys
import time
from functools import partial
from shutil import copy

import gadma
import moments

import Inference
from DemModelUpdater import Updater

SKIP_DIRS = ['__pycache__']

KNOWN_ALGORITHMS = {
    'my_bayes': lambda *args, **kwargs: KNOWN_ALGORITHMS['bayes'](*args, **kwargs, my=True),

    'bayes': lambda data, func, lower_bound, upper_bound, p_ids, iter_num, num_init_pts, filename, my=False:
    Inference.optimize_bayes(data, func,
                             lower_bound, upper_bound, p_ids,
                             max_iter=iter_num,
                             num_init_pts=num_init_pts,
                             output_log_file=filename,
                             my=my),

    'gadma': lambda data, func, lower_bound, upper_bound, p_ids, iter_num, num_init_pts, filename:
    gadma.Inference.optimize_ga(len(p_ids), data, func,
                                lower_bound=lower_bound,
                                upper_bound=upper_bound,
                                p_ids=p_ids,
                                optimization_name=None,
                                maxeval=iter_num,
                                num_init_pts=num_init_pts,
                                output_log_file=filename),

    'random_search': lambda data, func, lower_bound, upper_bound, p_ids, iter_num, num_init_pts, filename:
    gadma.Inference.optimize_ga(len(p_ids), data, func,
                                lower_bound=lower_bound,
                                upper_bound=upper_bound,
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


def run(data_dir, algorithm=None, start_idx=0, start_time=None, output_log_dir='results'):
    for dirpath, _, files in os.walk(data_dir):
        if any(x in dirpath for x in SKIP_DIRS + [output_log_dir]):
            continue

        result_dir = os.path.join(dirpath, output_log_dir, start_time)

        bench_log_path = os.path.join(result_dir, 'bench.log')
        cur_log_print = partial(log_print, bench_log_path)
        """
        Предполагаем, что данные хранятся так:
        
        data_dir_name
            -'data.fs'
            -'demographic_model.py'
            -'simulate_with_moments.py'
        """
        try:
            data_fs_file, model_file, _ = map(lambda x: os.path.join(dirpath, x), sorted(files))
        except ValueError:
            cur_log_print('Unsupported structure')
            continue

        dem_model = importlib.import_module(model_file.replace(os.path.sep, '.').rstrip('.py'))

        # TODO
        small_test_iter_num = 4
        small_num_init_pts = 2

        # Load the no_data
        data = moments.Spectrum.from_file(data_fs_file)

        # TODO: do it only when is output_log_dir a subdirectory of data_dir

        cur_log_print(f'Start {start_idx} for {algorithm} algorithm')

        optim = KNOWN_ALGORITHMS.get(algorithm)
        if optim is None:
            cur_log_print("Unknown algorithm")
        else:
            algorithm_dir = os.path.join(result_dir, algorithm)
            log_dir = os.path.join(algorithm_dir, f'start{start_idx}')
            os.makedirs(log_dir, exist_ok=True)

            t1 = time.time()
            optim(data, dem_model.model_func,
                  dem_model.lower_bound, dem_model.upper_bound, dem_model.p_ids,
                  small_test_iter_num, small_num_init_pts,
                  os.path.join(log_dir, 'evaluations.log'))
            t2 = time.time()
            cur_log_print(f'\tEvaluation time: {t2 - t1} ({start_idx} for {algorithm})')
        cur_log_print(f'Finish {start_idx} for {algorithm} algorithm')


def pre_run(data_dir, start_time, output_log_dir='results'):
    for dirpath, _, files in os.walk(data_dir):
        if any(x in dirpath for x in SKIP_DIRS + [output_log_dir]):
            continue

        data_fs_file, model_file, sim_file = map(lambda x: os.path.join(dirpath, x), sorted(files))

        # save version of code, which it was launched
        result_dir = os.path.join(dirpath, output_log_dir, start_time)
        os.makedirs(result_dir, exist_ok=True)
        copy('Benchmark.py', os.path.join(result_dir, 'CurrentVersionBenchmark.py'))

        sys.path.insert(0, dirpath)
        Updater(model_file).check_model(sim_file)


if __name__ == '__main__':
    start_time = time.strftime('%m.%d[%X]')
    # pre_run('data_4_DivMig', start_time)
    # run('data_4_DivMig', 'bayes', 0, start_time)
    # data_dirs = list(filter(lambda x: x.startswith('data'), next(os.walk('.'))[1]))
    data_dirs = ['data_2_DivMigr']
    # data_dirs = ['data_4_DivMig']
    [pre_run(d, start_time) for d in data_dirs]

    # algos = ['bayes', 'gadma', 'random_search']
    # algos = ['my_bayes', 'bayes']
    algos = ['bayes']

    num_starts = 2

    X = [(d, a, i, start_time) for d in data_dirs for a in algos for i in range(num_starts)]

    num_processes = 2
    pool = mp.Pool(num_processes)
    pool.map(partial(parallel_wrap, run), X)
    pool.close()
    pool.join()
