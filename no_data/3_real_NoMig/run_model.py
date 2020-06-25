import moments
import numpy as np
from demographic_model import model_func

# Parameters from Jouganous et al. (2017) paper (Table 4):
N_A = 11293
N_Af = 24486
N_B = 3034
N_Eu0 = 2587
r_Eu = 0.17e-2  # it is percent in table
N_As0 = 958
r_As = 0.30e-2  # it is percent in table
m_Af_B = 15.6e-5
m_Af_Eu = 1.00e-5
m_Af_As = 0.48e-5
m_Eu_As = 3.99e-5
T_Af = 349e3  # kya
T_B = 121e3  # kya
T_Eu_As = 44e3  # kya

# Time for one generation and mutation rate in paper
t_gen = 29  # years
mu = 1.44e-8

# translate these values to the one from model
# 1. Translate time to time of intervals and
T_Af -= T_B
T_B -= T_Eu_As
# Translate it to generations
T_g_Af = T_Af / t_gen
T_g_B = T_B / t_gen
T_g_Eu_As = T_Eu_As / t_gen
# 2. Get final sizes from r
# (1 + r) ** t = N_final / N_init
# N_final = N_init * ((1 + r) ** t)
# t is in generations!
N_Eu = N_Eu0 * ((1 + r_Eu) ** T_g_Eu_As)
N_As = N_As0 * ((1 + r_As) ** T_g_Eu_As)
print(N_Eu, N_As)
# 3. Get relative sizes
Npop = np.array([N_Af, N_B, N_Eu0, N_Eu, N_As0, N_As])
nuAf, nuB, nuEu0, nuEu, nuAs0, nuAs = Npop / N_A
# 4. Translate migrations by * 2 * N_A
# m = np.array([m_Af_B, m_Af_Eu, m_Af_As, m_Eu_As])
# mAfB, mAfEu, mAfAs, mEuAs = m * 2 * N_A
# 5. Translate time from generations by / (2 * N_A)h
TAf = T_g_Af / (2 * N_A)
TB = T_g_B / (2 * N_A)
TEuAs = T_g_Eu_As / (2 * N_A)
# Form params
params = (nuAf, nuB, nuEu0, nuEu, nuAs0, nuAs,
          TAf, TB, TEuAs)

# print(np.array2string(np.array(params), precision=7, separator=', ', max_line_width=np.inf))
# params = [2.168, 0.269, 0.229, 3.015, 0.085, 7.987, 0.348, 0.118, 0.067]
# params = [1.213, 0.459, 0.71,  0.879, 1.117, 0.185, 1.416, 0.089, 0.045] # gadma
# [3.571e+00 1.000e+02 3.184e+00 2.662e+00 1.138e+00 8.168e-02 5.000e+00 5.000e+00 6.059e-02] my_bayes(MPI)
params = [3.401e+00, 1.135e+00, 4.444e-01, 7.371e-01, 2.812e-02, 1.000e+02, 2.149e-02, 1.578e-01, 4.213e-02]

# Load data
data = moments.Spectrum.from_file("data.fs")

ns = (40, 40, 40)  # data.sample_sizes

# Draw model
from matplotlib import pyplot as plt

model = moments.ModelPlot.generate_model(model_func, params, (40, 40, 40))
moments.ModelPlot.plot_model(model,
                             fig_title='',
                             pop_labels=['YRI', 'CEU', 'CHB'],
                             nref=N_A,
                             draw_scale=True,
                             gen_time=0.029,
                             gen_time_units="Thousand years",
                             grid=False,
                             reverse_timeline=True)
plt.savefig('modelMPI.png', bbox_inches='tight')

# # Simulate
# t1 = time.time()
model = model_func(params, ns)
# t2 = time.time()
# print(f"Time of simulation is {(t2 - t1):2f} seconds.")
#
# # Calculate ll
ll = moments.Inference.ll_multinom(model, data)
print(f"Log likelihood of model is equal to {ll:2f}.")
