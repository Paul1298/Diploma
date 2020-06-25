import importlib
import os
import sys
import time
import random
from functools import partial
import multiprocessing as mp

import moments
import numpy as np

import Comparison


class Counter():
    def __init__(self, cur_time):
        self.cnt = 0
        self.cur_time = cur_time


def wrap(C, log_file, func, *args):
    if C.cnt == 100:
        raise Exception
    C.cnt += 1
    t = time.time()
    print(time.time() - C.cur_time, file=log_file, end='    ')
    C.cur_time = t
    log_file.flush()
    return func(*args)


def run_loc(data_dir, stdout, best_model_params, best_ll):
    data_fs_file, model_file = map(lambda x: os.path.join(data_dir, x), ['data.fs', 'demographic_model.py'])

    dem_model = importlib.import_module(model_file.replace(os.path.sep, '.').rstrip('.py'))
    func = dem_model.model_func

    # Load the no_data
    data = moments.Spectrum.from_file(data_fs_file)
    ns = data.sample_sizes

    print(best_model_params)
    print('Was log composite likelihood: {0}'.format(best_ll))

    popt = best_model_params
    start = time.time()
    C = Counter(start)
    print('Beginning optimization ************************************************')
    try:
        popt = moments.Inference.optimize_log(best_model_params, data, partial(wrap, C, stdout, func),
                                              lower_bound=dem_model.lower_bound,
                                              upper_bound=dem_model.upper_bound,
                                              verbose=1, maxiter=100)
    except Exception:
        print('Finished optimization **************************************************')

    print('Time: {}'.format(time.time() - start))
    print(popt)

    # Calculate the best-fit model AFS.
    model = func(popt, ns)
    # Likelihood of the no_data given the model AFS.
    ll_model = moments.Inference.ll_multinom(model, data)
    print('Maximum log composite likelihood: {0}'.format(ll_model))


def local(data_dir, start_dir):
    for dirpath, _, files in os.walk(start_dir):
        for f in files:
            log_file = os.path.join(dirpath, f)

            if f == 'evaluations.log':
                with open(log_file) as file:
                    next(file)

                    best_ll = -np.inf
                    for iter_num, line in enumerate(file):
                        if iter_num == 200:
                            break
                        parts = line.strip().split('\t')

                        logLL = -float(parts[1])

                        if logLL > best_ll:
                            best_ll = logLL
                            best_model_params = np.fromstring(parts[2][1:-1], dtype=float, sep=',')
                            this_line = line
                            # print(dirpath, iter_num + 2)

                    local_path = os.path.join(dirpath, 'local.log')
                    sys.stdout = open(local_path, 'w')
                    run_loc(data_dir, sys.stdout, best_model_params, best_ll)
                    sys.stdout.close()


if __name__ == '__main__':
    # data_dirs = ['data_4_DivNoMig']
    # date = '04/23/[23:49:38]'

    data_dirs = ['data_5_DivNoMig']
    date = '05/30/[14:54:30]'

    last_start_dir, dirnames = Comparison.get_dir(data_dirs[0], date)
    # dirnames = filter(lambda x: not x.startswith('_'), dirnames)
    dirnames = sorted(dirnames)
    dirnames = ['gadma', 'my_bayes(MPI)']
    X = []
        # ['data_5_DivNoMig/results/05/30/[14:54:30]/gadma/start11',
        #  'data_5_DivNoMig/results/05/30/[14:54:30]/gadma/start15']

    for i, algorithm in enumerate(sorted(dirnames)):
        algo_dir = os.path.join(last_start_dir,
                                algorithm)
        for dirpath, _, files in os.walk(algo_dir):
            last_dir = dirpath.split('/')[-1]
            if last_dir.startswith('start'):
                X.append(dirpath)


    random.shuffle(X)
    print(X)

    num_processes = 6
    pool = mp.Pool(num_processes)

    pool.map(partial(local, data_dirs[0]), X)
    pool.close()
    pool.join()
