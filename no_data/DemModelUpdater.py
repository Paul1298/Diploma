import importlib
from inspect import getsourcelines


class Updater:
    def __init__(self, filename: str):
        self.filename = filename
        self.model_func_name = 'model_func'
        self.dem_model = None  # init in check_model

    def __append_new_code(self, var_name, var_context):
        with open(self.filename, 'a') as f:
            f.write(f'\n\n{var_name} = {var_context}\n')

    def __append_p_ids(self):
        # Основано на предположении о том, что параметры распаковываются в 1й строке
        # как вариант можно поискать по подстроке '= params'

        # 0 because getsourcelines "Return a list of source lines and starting line number for an object"
        first_string = getsourcelines(self.dem_model.__getattribute__(self.model_func_name))[0][1]
        p_ids = [x.lower()[0] for x in first_string.split()[:-2]]
        self.__append_new_code('p_ids', p_ids)

    DEFAULT_BDS = {
        'n': [1e-2, 100],
        't': [0, 5],
        'm': [0, 10],
        's': [0, 1],
    }

    def __append_bounds(self, bound_type):
        bound = [self.DEFAULT_BDS[p][bound_type == 'upper'] for p in self.dem_model.p_ids]
        self.__append_new_code(bound_type + '_bound', bound)

    def check_model(self, sim_file):
        # TODO: add try except import wrapper
        self.dem_model = importlib.import_module(self.filename.replace('/', '.').rstrip('.py'))

        if self.model_func_name not in dir(self.dem_model):
            # inspect.isfunction
            raise Exception  # TODO: specify it

        exist_attrs = dir(self.dem_model)
        if 'p_ids' not in exist_attrs:
            self.__append_p_ids()

        importlib.reload(self.dem_model)

        if 'lower_bound' not in exist_attrs:
            self.__append_bounds('lower')

        if 'upper_bound' not in exist_attrs:
            self.__append_bounds('upper')

        if 'max_ll' not in exist_attrs:
            sim_mod = importlib.import_module(sim_file.replace('/', '.').rstrip('.py'))
            self.__append_new_code('max_ll', sim_mod.__getattribute__('ll_model'))

        importlib.reload(self.dem_model)

# if __name__ == '__main__':
# filename = 'data_example1/demographic_model.py'
# Updater(filename).check_model()
