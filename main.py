import birdepy as bd
import birdepy.gpu_functions as bdg
import numpy as np
import matplotlib.pyplot as plt
import time
from tabulate import tabulate
# from scipy.stats import wasserstein_distance as dist
from scipy.spatial.distance import jensenshannon as dist
import seaborn as sns
import tikzplotlib
plt.rcParams['figure.dpi']= 300


seed = 2021
gamma = 0.75
nu = 0.5
N = 100
param = [gamma, nu, 1/N, 0]
z0 = 10
zt = np.arange(0, 60, 1)
t = 10
model = 'Verhulst'

k = 10**4
times = np.arange(0,10.1,0.1)

path_exact = bd.simulate.discrete(param, model, z0, times, k, method="exact", seed=seed)
path_ea    = bd.simulate.discrete(param, model, z0, times, k, method="ea", seed=seed)
path_ma    = bd.simulate.discrete(param, model, z0, times, k, method="ma", seed=seed)
path_gwa   = bd.simulate.discrete(param, model, z0, times, k, method="gwa", seed=seed)

fig, axs = plt.subplots(1,1, figsize=(5,3))
axs.step(times, path_exact[0:3, :].T, where="post", color="tab:blue")
axs.plot(times, np.mean(path_exact, axis=0), color='k')
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
axs.set_title('exact')
axs.set_ylabel('$Z(t)$')
axs.set_xlabel('t')
tikzplotlib.save("path_exact.tex")

fig, axs = plt.subplots(1,1, figsize=(5,3))
axs.step(times, path_ea[0:3, :].T, where="post", color="tab:green")
axs.plot(times, np.mean(path_ea, axis=0), color='k')
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
axs.set_title('ea')
axs.set_ylabel('$Z(t)$')
axs.set_xlabel('t')
tikzplotlib.save("path_ea.tex")

fig, axs = plt.subplots(1,1, figsize=(5,3))
axs.step(times, path_ma[0:3, :].T, where="post", color="tab:red")
axs.plot(times, np.mean(path_ma, axis=0), color='k')
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
axs.set_title('ma')
axs.set_ylabel('$Z(t)$')
axs.set_xlabel('t')
tikzplotlib.save("path_ma.tex")

fig, axs = plt.subplots(1,1, figsize=(5,3))
axs.step(times, path_gwa[0:3, :].T, where="post", color="tab:purple")
axs.plot(times, np.mean(path_gwa, axis=0), color='k')
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
axs.set_title('gwa')
axs.set_ylabel('$Z(t)$')
axs.set_xlabel('t')
tikzplotlib.save("path_gwa.tex")


k = 10**4

zt = np.arange(0, 65, 1)

print("expm experiment 1a in progress                       ", end="\r")
tic = time.time()
prob_expm = bd.probability(z0, zt, t, param, model, z_trunc=[0, N])
toc_expm = time.time() - tic
print("expm experiment 1a completed  ")

print("exact (gpu) experiment 1a in progress               ", end="\r")
tic = time.time()
prob_bdg = bdg.probability(z0, zt, t, param, model, k=k, seed=seed)
toc_bdg = time.time() - tic
print("exact (gpu) experiment 1a completed  ")

print("exact (cpu) experiment 1a in progress               ", end="\r")
tic = time.time()
prob_exact = bd.probability(z0, zt, t, param, model, method="sim", k=k, sim_method="exact", seed=seed)
toc_exact = time.time() - tic
print("exact experiment 1a completed  ")

print("Euler experiment 1a in progress                     ", end="\r")
tic = time.time()
prob_ea = bd.probability(z0, zt, t, param, model, method="sim", k=k, sim_method="ea", seed=seed)
toc_ea = time.time() - tic
print("Euler experiment 1a completed  ")

print("midpoint experiment 1a in progress                  ", end="\r")
tic = time.time()
prob_ma = bd.probability(z0, zt, t, param, model, method="sim", k=k, sim_method="ma", seed=seed)
toc_ma = time.time() - tic
print("midpoint experiment 1a completed                    ")

print("linear experiment 1a in progress", end="\r")
tic = time.time()
prob_gwa = bd.probability(z0, zt, t, param, model, method="sim", k=k, sim_method="gwa", seed=seed)
toc_gwa = time.time() - tic
print("linear experiment 1a completed                      ")

print(tabulate([["bdg", "exact", "ea", "ma", "gwa"],
          ["Compute time (secs)", toc_bdg, toc_exact, toc_ea, toc_ma, toc_gwa]],
         headers="firstrow", floatfmt=".4f", tablefmt='latex'))


fig, ax = plt.subplots()
ax.plot(zt, prob_expm[0], '-', label="expm", color="gray")
ax.plot(zt, prob_exact[0], '+', label="exact (cpu)")
ax.plot(zt, prob_bdg[0], 'x', label="exact (gpu)")
ax.plot(zt, prob_ea[0], '*', label="ea", markersize=3)
ax.plot(zt, prob_ma[0], '^', label="ma", markersize=3)
ax.plot(zt, prob_gwa[0], 'o', label="gwa", markersize=3)
ax.legend(loc="upper right")
ax.set_xlabel('z')
ax.set_ylabel('P(Z(10)=z | Z(0) = 10)')
tikzplotlib.save("kde1a.tex")

k = 10**5

zt = np.arange(0, 65, 1)

print("expm experiment 1b in progress                       ", end="\r")
tic = time.time()
prob_expm = bd.probability(z0, zt, t, param, model, z_trunc=[0, N])
toc_expm = time.time() - tic
print("expm experiment 1b completed  ")

print("exact (gpu) experiment 1b in progress               ", end="\r")
tic = time.time()
prob_bdg = bdg.probability(z0, zt, t, param, model, k=k)
toc_bdg = time.time() - tic
print("exact (gpu) experiment 1b completed  ")

print("exact (cpu) experiment 1b in progress               ", end="\r")
tic = time.time()
prob_exact = bd.probability(z0, zt, t, param, model, method="sim", k=k, sim_method="exact")
toc_exact = time.time() - tic
print("exact experiment 1b completed  ")

print("Euler experiment 1b in progress                     ", end="\r")
tic = time.time()
prob_ea = bd.probability(z0, zt, t, param, model, method="sim", k=k, sim_method="ea")
toc_ea = time.time() - tic
print("Euler experiment 1b completed  ")

print("midpoint experiment 1b in progress                  ", end="\r")
tic = time.time()
prob_ma = bd.probability(z0, zt, t, param, model, method="sim", k=k, sim_method="ma")
toc_ma = time.time() - tic
print("midpoint experiment 1b completed                    ")

print("linear experiment 1b in progress", end="\r")
tic = time.time()
prob_gwa = bd.probability(z0, zt, t, param, model, method="sim", k=k, sim_method="gwa")
toc_gwa = time.time() - tic
print("linear experiment 1b completed                      ")

print(tabulate([["bdg", "exact", "ea", "ma", "gwa"],
          ["Compute time (secs)", toc_bdg, toc_exact, toc_ea, toc_ma, toc_gwa]],
         headers="firstrow", floatfmt=".4f", tablefmt='latex'))

fig, ax = plt.subplots()
ax.plot(zt, prob_expm[0], '-', label="expm", color="gray")
ax.plot(zt, prob_exact[0], '+', label="exact (cpu)")
ax.plot(zt, prob_bdg[0], 'x', label="exact (gpu)")
ax.plot(zt, prob_ea[0], '*', label="ea", markersize=3)
ax.plot(zt, prob_ma[0], '^', label="ma", markersize=3)
ax.plot(zt, prob_gwa[0], 'o', label="gwa", markersize=3)
ax.legend(loc="upper right")
ax.set_xlabel('z')
ax.set_ylabel('P(Z(10)=z | Z(0) = 10)')
tikzplotlib.save("kde1b.tex")

k = 10**4

N_vals = np.arange(10, 160, 10)

acc_exact_N = np.zeros(len(N_vals))
eff_exact_N = np.zeros(len(N_vals))
acc_gpu_N = np.zeros(len(N_vals))
eff_gpu_N = np.zeros(len(N_vals))
acc_ea_N = np.zeros(len(N_vals))
eff_ea_N = np.zeros(len(N_vals))
acc_ma_N = np.zeros(len(N_vals))
eff_ma_N = np.zeros(len(N_vals))
acc_gwa_N = np.zeros(len(N_vals))
eff_gwa_N = np.zeros(len(N_vals))


for idx, N in enumerate(N_vals):
    param = [gamma, nu, 1/N, 0]
    zt = np.arange(0, N+1, 1)
    prob_expm = bd.probability(z0, zt, t, param, model, z_trunc=[0, N])
    
    tic = time.time()
    prob_gpu_test = bdg.probability(z0, zt, t, param, model, k=k, seed=seed)
    eff_gpu_N[idx] = time.time() - tic
    acc_gpu_N[idx] = dist(prob_expm[0], prob_gpu_test[0])
    
    tic = time.time()
    prob_exact_test = bd.probability(z0, zt, t, param, model, method="sim", k=k, sim_method="exact", seed=seed)
    eff_exact_N[idx] = time.time() - tic
    acc_exact_N[idx] = dist(prob_expm[0], prob_exact_test[0])
    
    tic = time.time()
    prob_ea_test = bd.probability(z0, zt, t, param, model, method="sim", k=k,sim_method="ea", seed=seed)
    eff_ea_N[idx] = time.time() - tic
    acc_ea_N[idx] = dist(prob_expm[0], prob_ea_test[0])
    
    tic = time.time()
    prob_ma_test = bd.probability(z0, zt, t, param, model, method="sim", k=k, sim_method="ma", seed=seed)
    eff_ma_N[idx] = time.time() - tic
    acc_ma_N[idx] = dist(prob_expm[0], prob_ma_test[0])

    tic = time.time()
    prob_gwa_test = bd.probability(z0, zt, t, param, model, method="sim", k=k, sim_method="gwa", seed=seed)
    eff_gwa_N[idx] = time.time() - tic
    acc_gwa_N[idx] = dist(prob_expm[0], prob_gwa_test[0])
    print('N experiment ',100*(1+idx)/len(N_vals), '% complete                  ', end="\r")
    
    fig, axs = plt.subplots(1,1, figsize=(5,3))
axs.plot(N_vals, acc_exact_N, color='tab:blue', label="Alg. 1 (CPU)",linestyle="-")
axs.plot(N_vals, acc_gpu_N, color='tab:orange', label="Alg. 1 (CUDA)", linestyle=(0, (1,1)))
axs.plot(N_vals, acc_ea_N, color='tab:green', label="Alg. 2",linestyle="--")
axs.plot(N_vals, acc_ma_N, color='tab:red', label="Alg. 3",linestyle=(0, (1,10)))
axs.plot(N_vals, acc_gwa_N, color='tab:purple', label="Alg. 4",linestyle="-.")
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
axs.set_ylabel('Accuracy')
axs.set_xlabel('$N$')
axs.legend(loc="upper left")
tikzplotlib.save("acc_N.tex")

fig, axs = plt.subplots(1,1, figsize=(5,3))
axs.plot(N_vals, eff_exact_N, color='tab:blue', label="Alg. 1 (GPU)", linestyle="-")
axs.plot(N_vals, eff_gpu_N, color='tab:orange', label="Alg. 1 (CUDA)", linestyle=(0, (1,1)))
axs.plot(N_vals, eff_ea_N, color='tab:green', label="Alg. 2", linestyle="--")
axs.plot(N_vals, eff_ma_N, color='tab:red', label="Alg. 3", linestyle=(0, (1,10)))
axs.plot(N_vals, eff_gwa_N, color='tab:purple', label="Alg. 4", linestyle="-.")
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
axs.set_ylabel('Efficiency')
axs.set_xlabel('$N$')
tikzplotlib.save("eff_N.tex")

N = 100
param = [gamma, nu, 1/N, 0]
zt = np.arange(0, N+1, 1)

k_power_vals = np.arange(2, 5.5, 0.5)

acc_exact_k = np.zeros(len(k_power_vals))
eff_exact_k = np.zeros(len(k_power_vals))
acc_gpu_k = np.zeros(len(k_power_vals))
eff_gpu_k = np.zeros(len(k_power_vals))
acc_ea_k = np.zeros(len(k_power_vals))
eff_ea_k = np.zeros(len(k_power_vals))
acc_ma_k = np.zeros(len(k_power_vals))
eff_ma_k = np.zeros(len(k_power_vals))
acc_gwa_k = np.zeros(len(k_power_vals))
eff_gwa_k = np.zeros(len(k_power_vals))

prob_expm = bd.probability(z0, zt, t, param, model, z_trunc=[0, N])

for idx, k_pow in enumerate(k_power_vals):
    
    tic = time.time()
    prob_exact_test = bd.probability(z0, zt, t, param, model, method="sim", k=int(10**k_pow), sim_method="exact", seed=seed)
    eff_exact_k[idx] = time.time() - tic
    acc_exact_k[idx] = dist(prob_expm[0], prob_exact_test[0])

    tic = time.time()
    prob_gpu_test = bdg.probability(z0, zt, t, param, model, k=int(10**k_pow), seed=seed)
    eff_gpu_k[idx] = time.time() - tic
    acc_gpu_k[idx] = dist(prob_expm[0], prob_gpu_test[0])
    
    tic = time.time()
    prob_ea_test = bd.probability(z0, zt, t, param, model, method="sim", k=int(10**k_pow),sim_method="ea", seed=seed)
    eff_ea_k[idx] = time.time() - tic
    acc_ea_k[idx] = dist(prob_expm[0], prob_ea_test[0])
    
    tic = time.time()
    prob_ma_test = bd.probability(z0, zt, t, param, model, method="sim", k=int(10**k_pow), sim_method="ma", seed=seed)
    eff_ma_k[idx] = time.time() - tic
    acc_ma_k[idx] = dist(prob_expm[0], prob_ma_test[0])

    tic = time.time()
    prob_gwa_test = bd.probability(z0, zt, t, param, model, method="sim", k=int(10**k_pow), sim_method="gwa", seed=seed)
    eff_gwa_k[idx] = time.time() - tic
    acc_gwa_k[idx] = dist(prob_expm[0], prob_gwa_test[0])
    print('k experiment ', 100*(1+idx)/len(k_power_vals), '% complete                  ', end="\r")
  

fig, axs = plt.subplots(1,1, figsize=(5,3))
axs.plot(k_power_vals, acc_exact_k, color='tab:blue', label="Alg. 1 (CPU)",linestyle="-")
axs.plot(k_power_vals, acc_gpu_k, color='tab:orange', label="Alg. 1 (CUDA)",linestyle=(0, (1,1)))
axs.plot(k_power_vals, acc_ea_k, color='tab:green', label="Alg. 2",linestyle="--")
axs.plot(k_power_vals, acc_ma_k, color='tab:red', label="Alg. 3",linestyle=(0, (1,10)))
axs.plot(k_power_vals, acc_gwa_k, color='tab:purple', label="Alg. 4",linestyle="-.")
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
axs.set_ylabel('Accuracy')
axs.legend(loc="upper left")
axs.set_xlabel('$k$')
tikzplotlib.save("acc_k.tex")

fig, axs = plt.subplots(1,1, figsize=(5,3))
axs.plot(k_power_vals, eff_exact_k, color='tab:blue', label="Alg. 1 (CPU)", linestyle="-")
axs.plot(k_power_vals, eff_gpu_k, color='tab:orange', label="Alg. 1 (CUDA)", linestyle=(0, (1,1)))
axs.plot(k_power_vals, eff_ea_k, color='tab:green', label="Alg. 2", linestyle="--")
axs.plot(k_power_vals, eff_ma_k, color='tab:red', label="Alg. 3", linestyle=(0, (1,10)))
axs.plot(k_power_vals, eff_gwa_k, color='tab:purple', label="Alg. 4", linestyle="-.")
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
axs.set_ylabel('Efficiency')
axs.set_xlabel('$k$')
tikzplotlib.save("eff_k.tex")


N = 100
param = [gamma, nu, 1/N, 0]
zt = np.arange(0, N+1, 1)

tau_vals = np.arange(0.05, 0.55, 0.05)

acc_ea_tau = np.zeros(len(tau_vals))
eff_ea_tau = np.zeros(len(tau_vals))
acc_ma_tau = np.zeros(len(tau_vals))
eff_ma_tau = np.zeros(len(tau_vals))
acc_gwa_tau = np.zeros(len(tau_vals))
eff_gwa_tau = np.zeros(len(tau_vals))

prob_expm = bd.probability(z0, zt, t, param, model, z_trunc=[0, N])

for idx, tau in enumerate(tau_vals):

    tic = time.time()
    prob_ea_test = bd.probability(z0, zt, t, param, model, method="sim", k=10**4, sim_method="ea", tau=tau, seed=seed)
    eff_ea_tau[idx] = time.time() - tic
    acc_ea_tau[idx] = dist(prob_expm[0], prob_ea_test[0])
    
    tic = time.time()
    prob_ma_test = bd.probability(z0, zt, t, param, model, method="sim", k=10**4, sim_method="ma", tau=tau, seed=seed)
    eff_ma_tau[idx] = time.time() - tic
    acc_ma_tau[idx] = dist(prob_expm[0], prob_ma_test[0])

    tic = time.time()
    prob_gwa_test = bd.probability(z0, zt, t, param, model, method="sim", k=10**4, sim_method="gwa", tau=tau, seed=seed)
    eff_gwa_tau[idx] = time.time() - tic
    acc_gwa_tau[idx] = dist(prob_expm[0], prob_gwa_test[0])
    
    print('tau experiment ', 100*(1+idx)/len(tau_vals), '% complete                  ', end="\r")


fig, axs = plt.subplots(1,1, figsize=(5,3))
axs.plot(tau_vals, acc_ea_tau, color='tab:green', label="Alg. 2",linestyle="--")
axs.plot(tau_vals, acc_ma_tau, color='tab:red', label="Alg. 3",linestyle=(0, (1,3)))
axs.plot(tau_vals, acc_gwa_tau, color='tab:purple', label="Alg. 4",linestyle="-.")
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
axs.set_ylabel('Accuracy')
axs.set_xlabel('$\\tau$')
tikzplotlib.save("acc_tau.tex")

fig, axs = plt.subplots(1,1, figsize=(5,3))
axs.plot(tau_vals, eff_ea_tau, color='tab:green', label="Alg. 2", linestyle="--")
axs.plot(tau_vals, eff_ma_tau, color='tab:red', label="Alg. 3", linestyle=(0, (1,10)))
axs.plot(tau_vals, eff_gwa_tau, color='tab:purple', label="Alg. 4", linestyle="-.")
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
axs.set_ylabel('Efficiency')
axs.legend(loc="upper left")
axs.set_xlabel('$\\tau$')
tikzplotlib.save("eff_tau.tex")
