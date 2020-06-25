import time

import gadma
import numpy as np


class EvalLogger:
    def __init__(self, log_file, log):
        self.log_file = log_file
        self.log = log
        self.__write_caption()
        self.start_time = time.time()

    def __write_caption(self):
        with open(self.log_file, 'w') as log_file:
            print('Total time', 'logLL', 'model', 'iteration time', file=log_file, sep='\t')

    def log_wrapped(self, obj_func, *args, **kwargs):
        t1 = time.time()
        [[logLL]] = obj_func(*args, **kwargs)
        t2 = time.time()
        parameters = np.array(*args[0])
        parameters = np.array([np.exp(parameters.T[i]) if l else parameters.T[i] for i, l in enumerate(self.log)]).T
        X_str = np.array2string(parameters, precision=3, separator=', ', max_line_width=np.inf)
        gadma.support.write_to_file(self.log_file, t1 - self.start_time, logLL, X_str, t2 - t1)
        return logLL

    def write_calculated_values(self, p0s, lls):
        for p0, logLL in zip(p0s, lls):
            parameters = np.array(p0)
            parameters = np.array([np.exp(parameters.T[i]) if l else parameters.T[i] for i, l in enumerate(self.log)]).T
            X_str = np.array2string(parameters, precision=3, separator=', ', max_line_width=np.inf)
            gadma.support.write_to_file(self.log_file, 0, logLL, X_str, 0)
