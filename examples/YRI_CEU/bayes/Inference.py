import operator as op
from functools import partial

import GPy
import gadma
import numpy as np
from GPyOpt.methods import BayesianOptimization

import moments
from EvalLogger import EvalLogger


def generate_random_value(low_bound, upp_bound, identificator=None):
    """
    Generate random value for different parameters of models

    :param low_bound: Lower bound on parameter values.
    :param upp_bound: Upper bound on parameter values.
    :param identificator: Define parameter as
                            N - size of population, (in Nref units)
                            m - migration rate, (in 1/Nref units)
                            T - time,  (in Nref units)
                            s - split ratio. (in 1 units)

    Using https://github.com/ctlab/GADMA/blob/master/gadma/demographic_model.py#L385
    """
    if (identificator == 's') or \
            (low_bound == 0 and upp_bound == 1) \
            or not (low_bound <= 1 <= upp_bound):
        return np.random.uniform(low_bound, upp_bound)

    if low_bound <= 0:
        low_bound = 1e-15  # shift

    mode = 1.0

    # mean
    if low_bound >= mode:
        m = low_bound
    elif upp_bound <= mode:
        m = upp_bound
    else:
        m = mode
    # determine random function
    l = np.log(low_bound)
    u = np.log(upp_bound)
    m = np.log(m)

    if identificator == 't':
        random_generator = np.random.triangular
    else:
        random_generator = lambda a, b, c: gadma.support.sample_from_truncated_normal(b, max(b - a, c - b) / 3, a, c)

    # generate sample
    sample = random_generator(l, m, u)

    sample = np.exp(sample)
    return sample


def objective_func(model_func, data, ns, log, parameters_set):
    """
    Objective function for optimization.
    """
    lls = []
    for parameters in parameters_set:
        if log:
            parameters = np.exp(parameters)

        model = model_func(parameters, ns)

        # Likelihood of the data given the model AFS.
        ll_model = moments.Inference.ll_multinom(model, data)

        lls.append([-ll_model])  # minus for maximization
    return np.array(lls)


def check_to_stop(bo, iter_cnt, Y_eps):
    if hasattr(bo, 'Y_best'):
        if len(bo.Y_best) < iter_cnt:
            return True
        last_iters = bo.Y_best[-iter_cnt:]
        improvement = last_iters[0] - last_iters[-1]  # don't need abs because Y_best non increasing function
        if improvement < Y_eps:
            print('Improvement was %f, stop optimize' % improvement)
            return False
    return True


def optimize_bayes(data, model_func,
                   lower_bound, upper_bound, p_ids=None,
                   log=True,
                   max_iter=100,
                   num_init_pts=5,
                   kern_func_name=None,
                   output_log_file=None):
    """
    (using GA doc)
    Find optimized params to fit model to data using Bayesian Optimization.

    :param data: Spectrum with data.
    :param model_func: Function to evaluate model spectrum.
    :param lower_bound: Lower bound on parameter values. If using log, must be positive.
    :param upper_bound: Upper bound on parameter values. If using log, must be positive.
    :param p_ids: List of special symbols, that define parameters as N, m, T or s.
                    N - size of population, (in Nref units)
                    m - migration rate, (in 1/Nref units)
                    T - time,  (in Nref units)
                    s - split ratio. (in 1 units)

    :param log: If log = True, then model_func will be calculated for the logs of the parameters.
    :param max_iter: Maximum iterations to run for.
    :param num_init_pts: Number of initial points.
    :param kern_func_name: Name of the kernel that available at
                           https://github.com/SheffieldML/GPy/blob/devel/GPy/kern/src/stationary.py

                        You can see kernels using this code
                        ```
                        [m[0] for m in inspect.getmembers(GPy.kern.src.stationary, inspect.isclass)
                        if m[1].__module__ == 'GPy.kern.src.stationary']
                        ```
    :param output_log_file: Stream verbose log output into this filename. If None, no logging.
    :return:
    """
    if output_log_file:
        eval_log: EvalLogger = EvalLogger(output_log_file, log)

    ns = data.sample_sizes
    domain = np.array([{'name': 'var_' + str(i), 'type': 'continuous', 'domain': bd}
                       for i, bd in enumerate(zip(lower_bound, upper_bound))])
    kernel = op.attrgetter(kern_func_name)(GPy.kern.src.stationary)

    if p_ids is not None:
        p_ids = [x.lower()[0] for x in p_ids]
    else:
        p_ids = [None for _ in range(len(lower_bound))]

    p0 = np.array(
        [[generate_random_value(low_bound, upp_bound, p_id) for (low_bound, upp_bound, p_id)
          in zip(lower_bound, upper_bound, p_ids)]
         for _ in range(num_init_pts)])

    if log:
        shift = 1e-15
        shift_zero = lambda x: shift if x <= 0 else x
        for d in domain:
            dom = [shift_zero(bd) for bd in d['domain']]
            d.update({'domain': np.log(dom)})
        p0 = np.log(p0)

    obj_func = partial(objective_func, model_func, data, ns, log)
    if output_log_file:
        obj_func = partial(eval_log.log_wrapped, obj_func)
    bo = BayesianOptimization(f=obj_func,
                              domain=domain,
                              model_type='GP',
                              acquisition_type='EI',
                              kernel=kernel(input_dim=model_func.__code__.co_argcount),
                              ARD=True,  # By default, in kernel there's only one lengthscale:
                              # separate lengthscales for each dimension can be enables by setting ARD=True
                              X=p0,
                              )

    bo.run_optimization(max_iter=max_iter)

    return bo
