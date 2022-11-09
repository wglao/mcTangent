import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from jax import vmap, jit
import jax.numpy as jnp
from jax.example_libraries import stax

# get current parameters
from parameters_wave import *

# gound truth solution
truth = pd.read_csv('data/U_wave1d_test data_' + str(num_test_samples) + '_Nt_' + str(nt_test_data) + '_dt_test_' + str(dt_test) + '.csv')
truth = np.reshape(truth.to_numpy(), (num_test_samples, nt_test_data+1, N))

# data only
d_only = pickle.load('Network/wave1d_dt_train-test_' + str(dt) + '-' + str(dt_test) + '_seq_n_mc_' + str(n_seq_mc) +'_forward_mc_train_d' + str(num_train) + '_alpha_0_lr_' + str(learning_rate) + '_batch_' + str(batch_size) + '_nseq_' + str(n_seq) + '_layer_' + str(layers) + 'neurons' + str(units) + '_epochs_' + str(num_epochs))

# model-constrained
mc = pickle.load('Network/wave1d_dt_train-test_' + str(dt) + '-' + str(dt_test) + '_seq_n_mc_' + str(n_seq_mc) +'_forward_mc_train_d' + str(num_train) + '_alpha_' + str(1e5) + '_lr_' + str(learning_rate) + '_batch_' + str(batch_size) + '_nseq_' + str(n_seq) + '_layer_' + str(layers) + 'neurons' + str(units) + '_epochs_' + str(num_epochs))

# with noise
noisy = pickle.load('Network/wave1d_noise_0.02dt_train-test_' + str(dt) + '-' + str(dt_test) + '_seq_n_mc_' + str(n_seq_mc) +'_forward_mc_train_d' + str(num_train) + '_alpha_0_lr_' + str(learning_rate) + '_batch_' + str(batch_size) + '_nseq_' + str(n_seq) + '_layer_' + str(layers) + 'neurons' + str(units) + '_epochs_' + str(num_epochs))
# noisy = pickle.load('Network/wave1d_noise_0.02_dt_train-test_' + str(dt) + '-' + str(dt_test) + '_seq_n_mc_' + str(n_seq_mc) +'_forward_mc_train_d' + str(num_train) + '_alpha_0_lr_' + str(learning_rate) + '_batch_' + str(batch_size) + '_nseq_' + str(n_seq) + '_layer_' + str(layers) + 'neurons' + str(units) + '_epochs_' + str(num_epochs))

# model-constrained with noise
mc_n = pickle.load('Network/wave1d_noise_0.02dt_train-test_' + str(dt) + '-' + str(dt_test) + '_seq_n_mc_' + str(n_seq_mc) +'_forward_mc_train_d' + str(num_train) + '_alpha_' + str(1e5) + '_lr_' + str(learning_rate) + '_batch_' + str(batch_size) + '_nseq_' + str(n_seq) + '_layer_' + str(layers) + 'neurons' + str(units) + '_epochs_' + str(num_epochs))
# mc = pickle.load('Network/wave1d_noise_0.02_dt_train-test_' + str(dt) + '-' + str(dt_test) + '_seq_n_mc_' + str(n_seq_mc) +'_forward_mc_train_d' + str(num_train) + '_alpha_' + str(1e5) + '_lr_' + str(learning_rate) + '_batch_' + str(batch_size) + '_nseq_' + str(n_seq) + '_layer_' + str(layers) + 'neurons' + str(units) + '_epochs_' + str(num_epochs))

x = np.linspace(0, 1, N)

def ReLU(x):
    """ Rectified Linear Unit (ReLU) activation function """
    return jnp.maximum(0, x)

def Dense(inputs, W, b):
    return jnp.dot(inputs, W) + b

def forward_pass(params, u):
    W1, W2, b1, b2 = params
    u = Dense(ReLU(Dense(u, W1, b1)), W2, b2)
    return u

def single_forward_pass(params, un):
    u = un - facdt * dt * forward_pass(params, un)
    return u.flatten()

@jit
def neural_solver(params, U_test):

    u = U_test[0, :]

    U = jnp.zeros((nt_test_data + 1, N))
    U = U.at[0, :].set(u)

    for i in range(1, nt_test_data + 1):
        u = single_forward_pass(params, u)
        U = U.at[i, :].set(u)

    return U

neural_solver_batch = vmap(neural_solver, in_axes=(None, 0))

U_true = truth[0, :, :]
U_d_only = neural_solver_batch(d_only, truth)[0, :, :]
U_mc = neural_solver_batch(mc, truth)[0, :, :]
U_noisy = neural_solver_batch(noisy, truth)[0, :, :]
U_mcn = neural_solver_batch(mc_n, truth)[0, :, :]

fig = plt.figure()
plt.rcParams.update({'font.size': 22})
for i in range(n_plot):
        ut = jnp.reshape(U_true[Plot_Steps[i], :], (N, 1))
        ud = jnp.reshape(U_d_only[Plot_Steps[i], :], (N, 1))
        um = jnp.reshape(U_mc[Plot_Steps[i], :], (N, 1))
        un = jnp.reshape(U_noisy[Plot_Steps[i], :], (N, 1))
        umn = jnp.reshape(U_mc[Plot_Steps[i], :], (N, 1))
        ax = fig.add_subplot(1, 5, i+1)
        l1 = ax.plot(x, ut, '-', linewidth=2, label='True')
        l2 = ax.plot(x, ud, '--o', linewidth=2, label='Data only')
        l3 = ax.plot(x, um, '--x', linewidth=2, label='Model constrained (1e5)')
        l4 = ax.plot(x, un, '--v', linewidth=2, label='With noise (0.02)')
        l5 = ax.plot(x, umn, '--*', linewidth=2, label='Model constrained (1e5) and with noise (0.02)')
        ax.set_aspect('auto', adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('t = ' + str(Plot_Steps[i]))

        if i == 1:
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center')
