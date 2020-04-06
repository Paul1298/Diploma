import moments
from demographic_model import model_func as func

theta = 20000
popt = [1.0, 0.1, 5, 2.5, 0.05]
#p = [ 0.00419757,  0.99078189,  0.00207337,  0.05093703]
p = [ 0.9999,  0.1   ,  4.9995,  2.5008,  0.05  ]
p = [ 0.99996832,  0.09999811,  4.99997615,  2.50012035,  0.04999856]
p = [ 0.9991,  0.1   ,  5.0051,  2.4962,  0.05  ]

data = func(popt, [20,20]) * theta
model = data
ll_model = moments.Inference.ll_multinom(model, model)
print('Maximum log composite likelihood: {0}'.format(ll_model))

# model.to_file('data.fs')

# model = func(p, [20, 20])
# ll_model = moments.Inference.ll_multinom(model, data)
# print('Maximum log composite likelihood: {0}'.format(ll_model))
# theta = moments.Inference.optimal_sfs_scaling(model, data)
# print('Optimal value of theta / 2: {0}'.format(theta / 2))
# print([x * theta / 2 for x in p])
#
# print([x / theta * 2 for x in p])
# #
# #
#model *= theta
