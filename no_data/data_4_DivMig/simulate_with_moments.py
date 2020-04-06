import moments
from demographic_model import model_func as func

theta = 20000
# nu1, nu234, nu2, nu34, nu3, nu4, m1_234, m1_2, m1_34, m2_34, m1_3, m1_4, m2_3, m2_4, m3_4, T1, T2, T3
popt = [1.5, 0.8, 1.0, 0.5, 0.2, 0.3,   5, 0.5, 1, 3, 0.4, 0.3, 1.2, 1.3, 2,   0.1, 0.15, 0.05]

data = func(popt, [20, 20, 20, 20]) * theta
model = data

# model.to_file('data.fs')
# print("Data saved")
ll_model = moments.Inference.ll_multinom(model, model)
print('Maximum log composite likelihood: {0}'.format(ll_model))

