import importlib
import os

import gadma
import moments

import Inference
from examples.YRI_CEU.data.DemModelUpdater import Updater

if __name__ == '__main__':
    # TODO: update
    data_dir = 'data_example1'

    for dirpath, _, files in os.walk(data_dir):
        """
        Предполагаем, что данные хранятся так:
        
        data
            -'data.fs'
            -'demographic_model.py'
        """
        data_fs_file, model_file = map(lambda x: os.path.join(dirpath, x), sorted(files))

        Updater(model_file).check_model()

        dem_model = importlib.import_module(model_file.replace('/', '.').rstrip('.py'))

        small_test_iter_num = 2

        # Load the data
        data = moments.Spectrum.from_file(data_fs_file)
        Inference.optimize_bayes(data, dem_model.model_func,
                                 dem_model.lower_bound, dem_model.upper_bound, dem_model.p_ids,
                                 max_iter=small_test_iter_num,
                                 output_log_file=os.path.join(dirpath, 'BO_evaluations.log'))

        gadma.Inference.optimize_ga(len(dem_model.p_ids), data, dem_model.model_func,
                                    size_of_generation_in_ga=10,
                                    lower_bound=dem_model.lower_bound,
                                    upper_bound=dem_model.upper_bound,
                                    optimization_name=None,
                                    maxeval=small_test_iter_num,
                                    num_init_pts=5,
                                    output_log_file=os.path.join(dirpath, 'GA_evaluations.log'),
                                    )
