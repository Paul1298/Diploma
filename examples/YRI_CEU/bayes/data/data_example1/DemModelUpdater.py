import importlib.util
from inspect import getsourcelines


def check_model(filename):
    # TODO add try except import wrapper
    spec = importlib.util.spec_from_file_location('dem_model', filename)
    dem_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dem_model)

    model_func_name = 'model_func'
    if model_func_name not in dir(dem_model):
        # inspect.isfunction
        raise Exception  # TODO specify it
    if 'p_ids' not in dir(dem_model):
        # Основано на предположении о том, что параметры распаковываются в 1й строке
        # как вариант можно поискать по подстроке '= params'
        first_string = getsourcelines(dem_model.__getattribute__(model_func_name))[0][1]
        p_ids = [x.lower()[0] for x in first_string.split()[:-2]]
        with open(filename, 'a') as f:
            f.write(f'\n\np_ids = {p_ids}')


if __name__ == '__main__':
    # filename = input()
    filename = 'demographic_model.py'
    check_model(filename)
