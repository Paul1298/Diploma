import importlib
import os
import time

import moments
import numpy as np

import Comparison


def run_loc(data_dir, best_model_params):
    data_fs_file, model_file = map(lambda x: os.path.join(data_dir, x), ['data.fs', 'demographic_model.py'])

    dem_model = importlib.import_module(model_file.replace(os.path.sep, '.').rstrip('.py'))
    func = dem_model.model_func

    # Load the no_data
    data = moments.Spectrum.from_file(data_fs_file)
    ns = data.sample_sizes

    start = time.time()
    print('Beginning optimization ************************************************')
    popt = moments.Inference.optimize_log(best_model_params, data, func,
                                          lower_bound=dem_model.lower_bound,
                                          upper_bound=dem_model.upper_bound,
                                          verbose=1, maxiter=100)
    print('Finished optimization **************************************************')
    print('Time: {}'.format(time.time() - start))
    print(popt)

    # Calculate the best-fit model AFS.
    model = func(popt, ns)
    # Likelihood of the no_data given the model AFS.
    ll_model = moments.Inference.ll_multinom(model, data)
    print('Maximum log composite likelihood: {0}'.format(ll_model))

    model = func(best_model_params, ns)
    ll_model = moments.Inference.ll_multinom(model, data)
    print('Was log composite likelihood: {0}'.format(ll_model))


def local(data_dir, date):
    last_start_dir, dirnames = Comparison.get_dir(data_dir, date)
    # dirnames = filter(lambda x: not x.startswith('_'), dirnames)
    dirnames = sorted(dirnames)
    dirnames = ['gadma', 'my_bayes(MPI)']

    for i, algorithm in enumerate(sorted(dirnames)):
        algo_dir = os.path.join(last_start_dir,
                                algorithm)
        best_ll = -np.inf
        print(algo_dir)
        for dirpath, _, files in os.walk(algo_dir):
            for f in files:
                log_file = os.path.join(dirpath, f)

                if f == 'evaluations.log':
                    with open(log_file) as file:
                        next(file)

                        for iter_num, line in enumerate(file):
                            parts = line.strip().split('\t')

                            logLL = -float(parts[1])

                            if logLL > best_ll:
                                best_ll = logLL
                                best_model_params = np.fromstring(parts[2][1:-1], dtype=float, sep=',')
                                this_line = line
                                print(dirpath, iter_num + 2)
        print(algorithm, this_line)
        # run_loc(data_dir, best_model_params)


if __name__ == '__main__':
    data_dirs = ['data_4_DivNoMig']
    date = '04/23/[23:49:38]'

    # data_dirs = ['data_5_DivNoMig']
    # date = '05/30/[14:54:30]'

    [local(d_d, date) for d_d in data_dirs]
