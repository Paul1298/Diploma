import importlib
import multiprocessing as mp
import os
import sys
import time

import gadma
import moments

import Inference
from DemModelUpdater import Updater

SKIP_DIRS = ['__pycache__']

KNOWN_ALGORITHMS = {
    'bayes': lambda data, func, lower_bound, upper_bound, p_ids, iter_num, num_init_pts, filename:
    Inference.optimize_bayes(data, func,
                             lower_bound, upper_bound, p_ids,
                             max_iter=iter_num,
                             num_init_pts=num_init_pts,
                             output_log_file=filename),
    'gadma': lambda data, func, lower_bound, upper_bound, p_ids, iter_num, num_init_pts, filename:
    gadma.Inference.optimize_ga(len(p_ids), data, func,
                                lower_bound=lower_bound,
                                upper_bound=upper_bound,
                                p_ids=p_ids,
                                optimization_name=None,
                                maxeval=iter_num,
                                num_init_pts=num_init_pts,
                                output_log_file=filename)
}


def parallel_wrap(args):
    print(args)
    return run(*args)


def run(data_dir, algorithms=None, start_idx=0, num_cores=1, output_log_dir='results'):
    if type(algorithms) is str:
        algorithms = [algorithms]

    if algorithms is None:
        algorithms = []

    for dirpath, _, files in os.walk(data_dir):
        if any(x in dirpath for x in SKIP_DIRS + [output_log_dir]):
            continue

        """
        Предполагаем, что данные хранятся так:
        
        data_dir_name
            -'no_data.fs'
            -'demographic_model.py'
        """
        try:
            data_fs_file, model_file = map(lambda x: os.path.join(dirpath, x), sorted(files))
        except ValueError:
            print('Unsupported structure', file=sys.stderr)
            continue

        Updater(model_file).check_model()

        dem_model = importlib.import_module(model_file.replace(os.path.sep, '.').rstrip('.py'))

        # TODO
        small_test_iter_num = 1
        small_num_init_pts = 5

        # Load the no_data
        data = moments.Spectrum.from_file(data_fs_file)

        # TODO: do it only when is output_log_dir a subdirectory of data_dir
        result_dir = os.path.join(dirpath, output_log_dir, time.strftime('%m.%d %X'))

        print(f'Start{start_idx} for {algorithms} configuration', file=sys.stderr)
        start_dir = os.path.join(result_dir, f'start{start_idx}')

        for algorithm in algorithms:
            optim = KNOWN_ALGORITHMS.get(algorithm)
            if optim is None:
                print("Unknown algorithm", file=sys.stderr)
                # raise Exception("Unknown algorithm")
            else:
                print(f'\tEvaluation for {algorithm} start', file=sys.stderr)
                log_dir = os.path.join(start_dir, algorithm)
                os.makedirs(log_dir, exist_ok=True)

                t1 = time.time()
                optim(data, dem_model.model_func,
                      dem_model.lower_bound, dem_model.upper_bound, dem_model.p_ids,
                      small_test_iter_num, small_num_init_pts,
                      os.path.join(log_dir, 'evaluations.log'))
                t2 = time.time()
                print(f'\tEvaluation time: {t2 - t1}', file=sys.stderr)
            print(f'\tEvaluation for {algorithm} done', file=sys.stderr)
        print(f'Finish{start_idx} for {algorithms} configuration', file=sys.stderr)


if __name__ == '__main__':
    data_dirs = list(filter(lambda x: not x.startswith('__'), next(os.walk('.'))[1]))
    algos = ['bayes', 'gadma']
    num_starts = 2

    X = [(d, a, i) for d in data_dirs for a in algos for i in range(num_starts)]
    print(X)

    num_processes = 16
    # pool = mp.Pool(num_processes)
    pool = mp.Pool()
    res = pool.map(parallel_wrap, X)
    pool.close()
    pool.join()