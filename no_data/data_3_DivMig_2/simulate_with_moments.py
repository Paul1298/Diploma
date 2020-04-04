import moments
from demographic_model import model_func as func

theta = 20000
popt = [1.5, 0.5, 1.0, 0.5, 1.0, 3.0, 0.1, 0.05]
p0 = [  1.59842437e+00,   4.98573225e-01,   1.34974238e+00,
         3.54314463e-09,   1.96651803e-28,   1.95044681e-33,
         7.55549964e-02,   3.96051969e-02]
p0 = [1.4976502379213381, 0.5016337658846286, 0.99565773297692173, 0.4941762111481659, 1.0139167629096981, 2.9793492884082946, 0.1003714483406991, 0.049882924125576898]
p0 = [ 1.4992856 ,  0.49868817,  0.99942435,  0.50069398,  0.99700908,
        3.01551221,  0.09967205,  0.04998882]

data = func(popt, [20, 20, 20]) * theta
model = func(p0, [20, 20, 20])
ll_model = moments.Inference.ll_multinom(model, data)
print(('Maximum log composite likelihood: {0}'.format(ll_model)))
print(moments.Inference.ll_multinom(data, data))
theta = moments.Inference.optimal_sfs_scaling(model, data)
theta /= 2
print(('N_A:' + str(theta)))
import numpy as np
print(np.array(p0) * theta)
print(np.array(p0) / theta)
#model.to_file('simulated_data/00.fs')
