import importlib.util
from inspect import getsourcelines


class Updater:
    def __init__(self, filename):
        self.filename = filename
        self.model_func_name = 'model_func'

    @staticmethod
    def __append_new_code(name, context):
        with open(filename, 'a') as f:
            f.write(f'\n\n{name} = {context}\n')

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

    def check_model(self):
        # TODO: add try except import wrapper
        spec = importlib.util.spec_from_file_location('dem_model', filename)
        self.dem_model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.dem_model)

        if self.model_func_name not in dir(self.dem_model):
            # inspect.isfunction
            raise Exception  # TODO: specify it

        exist_attrs = dir(self.dem_model)
        if 'p_ids' not in exist_attrs:
            self.__append_p_ids()

            # чтобы если границ тоже нет, то подгрузить модуль с уже существующими p_ids
            spec.loader.exec_module(self.dem_model)

        if 'lower_bound' not in exist_attrs:
            self.__append_bounds('lower')

        if 'upper_bound' not in exist_attrs:
            self.__append_bounds('upper')


if __name__ == '__main__':
    # filename = input()
    filename = 'demographic_model.py'
    Updater(filename).check_model()
