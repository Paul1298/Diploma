# coding=utf-8
import GPy
import gadma
import numpy as np
from GPyOpt.methods import BayesianOptimization


def generate_random_value(low_bound, upp_bound):
    """
    Generate random value for different parameters of models

    Using https://github.com/ctlab/GADMA/blob/master/gadma/demographic_model.py#L385
    """
    if (low_bound == 0 and upp_bound == 1) \
            or not (low_bound <= 1 <= upp_bound):
        return np.random.uniform(low_bound, upp_bound)

    if low_bound <= 0:
        low_bound = 1e-15

    mode = 1.0

    # remember bounds and mean TODO: спросить Катю зачем это
    l = low_bound  # left bound
    u = upp_bound  # right bound
    # mean
    if low_bound >= mode:
        m = low_bound
    elif upp_bound <= mode:
        m = upp_bound
    else:
        m = mode
    # determine random function
    l = np.log(l)
    u = np.log(u)
    m = np.log(m)

    random_generator = lambda a, b, c: gadma.support.sample_from_truncated_normal(b, max(b - a, c - b) / 3, a, c)
    # generate sample
    sample = random_generator(l, m, u)

    sample = np.exp(sample)
    return sample


def optimize_bayes(number_of_params,
                   data, model_func,
                   log=True,
                   lower_bound=None, upper_bound=None,
                   max_iter=100,
                   pts=5,
                   p0=None):
    """
    (using GA doc)
    Find optimized params to fit model to data using Bayesian Optimization.

    :param number_of_params: Number of parameters to find.
    :param data: Spectrum with data.
    :param model_func: Function to evaluate model spectrum.
    :param log: If log = True, then model_func will be calculated for the logs of the parameters;
    :param lower_bound: Lower bound on parameter values. If not None, must be of
                        length equal to number_of_params.
    :param upper_bound: Upper bound on parameter values. If not None, must be of
                        length equal to number_of_params.
    :param max_iter: Maximum iterations to run for.
    :param pts: Number of initial points.
    :param p0: Initial parameters. You can start from some known parameters.

    :return:
    """
    ns = data.sample_sizes

    domain = np.array([{'name': 'var_' + str(i), 'type': 'continuous', 'domain': bd}
                       for i, bd in enumerate(zip(lower_bound, upper_bound))])

    p0 = np.array(
        [[generate_random_value(low_bound, upp_bound) for (low_bound, upp_bound) in zip(lower_bound, upper_bound)]
         for _ in range(pts)])

    if log:
        shift = 1e-15
        # domain_log_func = lambda d: if
        [d.update({'domain': np.log(d['domain'])}) for d in domain]
        p0 = np.log(p0)

    bo = BayesianOptimization(f=model_func,
                              domain=domain,
                              model_type='GP',
                              kernel=GPy.kern.Matern52(input_dim=ns),
                              acquisition_type='EI',
                              X=p0,
                              ARD=True
                              )
    return
