import importlib
import os

import gadma
import moments

import Inference
from examples.YRI_CEU.no_data.DemModelUpdater import Updater

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
                                size_of_generation_in_ga=10,  # TODO уточнить
                                optimization_name=None,
                                maxeval=iter_num,
                                num_init_pts=num_init_pts,
                                output_log_file=filename)
}


# TODO: change default num_starts to 20
def run(data_dir, num_cores=1, algorithms=None, output_log_dir='results', num_starts=2):
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
            print(dirpath)
            break

        Updater(model_file).check_model()

        dem_model = importlib.import_module(model_file.replace(os.path.sep, '.').rstrip('.py'))

        small_test_iter_num = 1
        small_num_init_pts = 5

        # Load the no_data
        data = moments.Spectrum.from_file(data_fs_file)

        # TODO: do it only when is output_log_dir a subdirectory of data_dir
        result_dir = os.path.join(dirpath, output_log_dir)
        os.makedirs(result_dir, exist_ok=True)

        # TODO: parallel
        for i in range(num_starts):
            # print(f'Start{i} for {algorithms} configuration')
            start_dir = os.path.join(result_dir, f'start{i}')
            os.mkdir(start_dir)

            for algorithm in algorithms:
                optim = KNOWN_ALGORITHMS.get(algorithm)
                if optim is None:
                    print("Unknown algorithm")
                    # raise Exception("Unknown algorithm")
                else:
                    # print(f'\tEvaluation for {optim} start')
                    log_dir = os.path.join(start_dir, algorithm)
                    os.mkdir(log_dir)

                    optim(data, dem_model.model_func,
                          dem_model.lower_bound, dem_model.upper_bound, dem_model.p_ids,
                          small_test_iter_num, small_num_init_pts,
                          os.path.join(log_dir, 'evaluations.log'))
                # print(f'\tEvaluation for {optim} done')
            # print(f'Finish{i} for {algorithms} configuration')


if __name__ == '__main__':
    run('data_example1', 1, ('bayes', 'gadma'))
