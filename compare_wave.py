import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import jax
from jax import vmap, jit
import jax.numpy as jnp
from jax.example_libraries import stax, optimizers

# get current parameters
from parameters_wave import *

x = np.linspace(0, 1, N)

# gound truth solution
truth = pd.read_csv('data/U_wave1d_test_data_' + str(num_test_samples) + '_Nt_' + str(nt_test_data) + '_dt_test_' + str(dt_test) + '.csv')
truth = np.reshape(truth.to_numpy(), (num_test_samples, nt_test_data+1, N))

# randomized initial condition
input_noise = False
if input_noise:
    ns, nt, nx = truth.shape
    nosie_vec = jax.random.normal(key_data_noise, truth.shape)
    noise_level = 0.02
    truth_noise = np.zeros(truth.shape)

    for i in range(ns):
        for j in range(nt):
                truth_noise[i,j,:] = truth[i,j,:] + noise_level * nosie_vec[i,j,:] * np.max(truth[i,j,:])
    
    plt.plot(x, truth[0,0,:])
    plt.plot(x, truth_noise[0,0,:])
    plt.show()

_, _, opt_get_params = optimizers.adam(learning_rate)
def unpickle_params(filepath):
    ret = pickle.load(open(filepath, 'rb'))
    ret = optimizers.pack_optimizer_state(ret)
    return opt_get_params(ret)

# data only
d_only_params = unpickle_params('Network/Best_wave1d_dt_train-test_' + str(dt) + '-' + str(dt_test) + '_seq_n_mc_' + str(n_seq_mc) +'_forward_mc_train_d' + str(num_train) + '_alpha_0_lr_' + str(learning_rate) + '_batch_' + str(batch_size) + '_nseq_' + str(n_seq) + '_layer_' + str(layers) + 'neurons' + str(units) + '_epochs_' + str(num_epochs))

# model-constrained
mc_params = unpickle_params('Network/Best_wave1d_dt_train-test_' + str(dt) + '-' + str(dt_test) + '_seq_n_mc_' + str(n_seq_mc) +'_forward_mc_train_d' + str(num_train) + '_alpha_' + str(1e5) + '_lr_' + str(learning_rate) + '_batch_' + str(batch_size) + '_nseq_' + str(n_seq) + '_layer_' + str(layers) + 'neurons' + str(units) + '_epochs_' + str(num_epochs))

# with noise
noisy_params = unpickle_params('Network/Best_wave1d_noise_0.02_dt_train-test_' + str(dt) + '-' + str(dt_test) + '_seq_n_mc_' + str(n_seq_mc) +'_forward_mc_train_d' + str(num_train) + '_alpha_0_lr_' + str(learning_rate) + '_batch_' + str(batch_size) + '_nseq_' + str(n_seq) + '_layer_' + str(layers) + 'neurons' + str(units) + '_epochs_' + str(num_epochs))

# model-constrained with noise
mcn_params = unpickle_params('Network/Best_wave1d_noise_0.02dt_train-test_' + str(dt) + '-' + str(dt_test) + '_seq_n_mc_' + str(n_seq_mc) +'_forward_mc_train_d' + str(num_train) + '_alpha_' + str(1e5) + '_lr_' + str(learning_rate) + '_batch_' + str(batch_size) + '_nseq_' + str(n_seq) + '_layer_' + str(layers) + 'neurons' + str(units) + '_epochs_' + '50000')


from network_dense_wave import *
# def ReLU(x):
#     """ Rectified Linear Unit (ReLU) activation function """
#     return jnp.maximum(0, x)

# def Dense(inputs, W, b):
#     return jnp.dot(inputs, W) + b

# def forward_pass(params, u):
#     W1, W2, b1, b2 = params
#     u = Dense(ReLU(Dense(u, W1, b1)), W2, b2)
#     return u

# def single_forward_pass(params, un):
#     u = un - facdt * dt * forward_pass(params, un)
#     return u.flatten()

# @jit
# def neural_solver(params, U_test):

#     u = U_test[0, :]

#     U = jnp.zeros((nt_test_data + 1, N))
#     U = U.at[0, :].set(u)

#     for i in range(1, nt_test_data + 1):
#         u = single_forward_pass(params, u)
#         U = U.at[i, :].set(u)

#     return U

# neural_solver_batch = vmap(neural_solver, in_axes=(None, 0))

plot_sample = 75
U_true = truth[plot_sample, :, :]
if input_noise:
    U_d_only = neural_solver_batch(d_only_params, truth_noise)[plot_sample, :, :]
    U_mc = neural_solver_batch(mc_params, truth_noise)[plot_sample, :, :]
    U_noisy = neural_solver_batch(noisy_params, truth_noise)[plot_sample, :, :]
    U_mcn = neural_solver_batch(mcn_params, truth_noise)[plot_sample, :, :]
else:
    U_d_only = neural_solver_batch(d_only_params, truth)[plot_sample, :, :]
    U_mc = neural_solver_batch(mc_params, truth)[plot_sample, :, :]
    U_noisy = neural_solver_batch(noisy_params, truth)[plot_sample, :, :]
    U_mcn = neural_solver_batch(mcn_params, truth)[plot_sample, :, :]

fontsize = 8
fig = plt.figure(figsize=((n_plot+1)*fontsize,fontsize))
plt.rcParams.update({'font.size': fontsize})
for i in range(n_plot):
        ut = jnp.reshape(U_true[Plot_Steps[i], :], (N, 1))
        ud = jnp.reshape(U_d_only[Plot_Steps[i], :], (N, 1))
        um = jnp.reshape(U_mc[Plot_Steps[i], :], (N, 1))
        un = jnp.reshape(U_noisy[Plot_Steps[i], :], (N, 1))
        umn = jnp.reshape(U_mc[Plot_Steps[i], :], (N, 1))
        
        ax = fig.add_subplot(1, n_plot, i+1)
        l1 = ax.plot(x, ut, '-', linewidth=3, label='True')
        l2 = ax.plot(x, ud, ':o', markevery=5, fillstyle='none', linewidth=3, label='Data only')
        l3 = ax.plot(x, um, ':v', markevery=5, fillstyle='none', linewidth=3, label='Model constrained (1e5)')
        l4 = ax.plot(x, un, ':x', markevery=5, linewidth=3, label='With noise (0.02)')
        l5 = ax.plot(x, umn, ':+', markevery=5, linewidth=3, label='Model constrained (1e5) and with noise (0.02)')

        # ax.set_aspect('auto', adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('t = ' + str(Plot_Steps[i]))

        if i == n_plot-1:
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='center right')

if input_noise:
    plt.savefig('figs/compare_wave_noise.png', bbox_inches='tight')
else:
    plt.savefig('figs/compare_wave.png', bbox_inches='tight')
# plt.show()
