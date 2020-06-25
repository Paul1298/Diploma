import importlib
import itertools
import multiprocessing as mp
import os
from functools import partial

import moments
import matplotlib.pyplot as plt
import numpy as np

import Benchmark
import Inference
from EvalLogger import EvalLogger


def calc(data_dir):
    # algo_map = {}
    # for dirpath, dirnames, files in os.walk(data_dir):
    #     if dirnames and dirnames[0].startswith('start'):
    #         cur_algo = dirpath.split('/')[-1]
    files = ['time.log']#, 'time2.log']
    times = []
    for f in files:
        log_file = os.path.join(data_dir, f)
        with open(log_file) as file:
            next(file)

            for iter_num, line in enumerate(file):
                parts = line.strip().split('\t')

                iter_time = float(parts[-1])

                times.append(iter_time)

    # print(times)

    fig1, ax1 = plt.subplots()
    ax1.boxplot(times, vert=False)
    fig1.savefig(os.path.join(data_dir, 'iter_time.png'), bbox_inches='tight')

    # with open(os.path.join(data_dir, data_dir.lstrip('data_') + '_time.tsv'), 'w') as log_file:
    #     print('min',
    #           'mean',
    #           'max',
    #           'count',
    #           file=log_file, sep='\t')

        # for a, v in .items():
        #     print(a,
        #           np.min(v),
        #           np.mean(v),
        #           np.max(v),
        #           len(v),
        #           file=log_file, sep='\t')


def time_from_random(data_dir):
    for dirpath, dirs, files in os.walk(data_dir):
        if any(x in dirpath for x in Benchmark.SKIP_DIRS + ['results']):
            dirs[:] = []
            continue
        data_fs_file, model_file = map(lambda x: os.path.join(dirpath, x), ['data.fs', 'demographic_model.py'])

        dem_model = importlib.import_module(model_file.replace(os.path.sep, '.').rstrip('.py'))

        p0 = np.array(
            [[Inference.generate_random_value(low_bound, upp_bound, 't') for (low_bound, upp_bound)
              in zip(dem_model.lower_bound, dem_model.upper_bound)]
             for _ in range(100)])

        p0 = np.log(p0)
        print(p0)
        print('ok')

        data = moments.Spectrum.from_file(data_fs_file)
        ns = data.sample_sizes
        log = [True for _ in range(len(dem_model.lower_bound))]

        eval_log: EvalLogger = EvalLogger(os.path.join(dirpath, 'time.log'), log)

        obj_func = partial(Inference.objective_func, dem_model.model_func, data, ns, log)
        obj_func = [partial(eval_log.log_wrapped, obj_func) for _ in range(len(p0))]
        # obj_func = [fc for _ in range(len(p0))]

        res = list(zip(obj_func, p0))
        return res


# def fc(x):
#     print(3)
#     return x


def apply(args):
    return args[0]([args[1]])


# def

if __name__ == '__main__':
    # # data_dirs = list(filter(lambda x: x.startswith('data'), next(os.walk('.'))[1]))
    # data_dirs = ['data_5_DivNoMig', '4_YRI_CEU_CHB_JPT_17_Jou',
    #              'data_4_DivNoMig', '4_real_NoMig']
    data_dirs = ['data_4_DivNoMig']
    [calc(d_d) for d_d in data_dirs]

    # # print(time_from_random(d_d) for d_d in data_dirs)
    # lsts = [time_from_random(d_d) for d_d in data_dirs]
    # X = [x for x in itertools.chain(*lsts)]
    # print(len(X))
    #
    # num_processes = 16
    # pool = mp.Pool(num_processes)
    #
    # pool.map(apply, X)
    # pool.close()
    # pool.join()
