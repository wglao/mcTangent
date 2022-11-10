# %%
# %%
import jax
from jax.example_libraries import stax, optimizers
import jax.numpy as jnp
from jax import grad, value_and_grad, vmap, random, jit, lax

from jax.config import config
import jax.numpy as jnp

from jax.nn.initializers import glorot_normal, normal, zeros, ones

import pandas as pd
import numpy as np

from scipy import sparse

import time

config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
from matplotlib import cm  # Colour map
import matplotlib.animation as animation

#! 1 - Spectral method for 2D Navier-Stoke equation initialize parameters
# initialize parameters
import parameters_wave
from parameters_wave import *

# ## 2 - Draw initial condition
#! 2.1 Saving the eigenvector and eigenvalues
# df_Basis = pd.read_csv('data/Basis_Modes.csv')
# Modes_100 = np.reshape(df_Basis.to_numpy(), (N, 100*N))

# Basis = Modes_100[:, :num_truncated_series]

# Solution_samples_array = pd.DataFrame({'Basis': Basis.flatten()})
# Solution_samples_array.to_csv('data/Basis.csv', index=False)

# %%
def single_step_forward_solver(un, dt):
    # use different difference schemes for edge case
    lu = len(un)
    u = un + (- dt / dx * (jnp.roll(un, -1) - jnp.roll(un, 1)) / 2 )
    uleft = un[0] + (dt / dx / 2 * (3*un[0] - 4*un[1] + un[2]))
    uright = un[lu-1] + (dt / -dx / 2 * (3*un[lu-1] - 4*un[lu-2] + un[lu-3]))
    u = u.at[0].set(uleft)
    u = u.at[lu-1].set(uright)
    return u

def generate_data_solver(IBC, dt, Nt, n):
    U_save = np.zeros((Nt+1, n)) # Nt + 1 for including the intial data

    u = jnp.reshape(IBC, (n,1))

    U_save[0, :] = u.flatten()

    for i in  range(1, Nt+1):
        u = single_step_forward_solver(u, dt)
        U_save[i, :] = u.flatten()

    # Compare solutions
    # fig = plt.figure(figsize=(32,10))
    # fig.patch.set_facecolor('xkcd:white')

    # plt.close()
    # ax = fig.add_subplot(1, 2, 1)
    # l1 = ax.plot(x, U_save[0,:], '-', label='True')
    # ax.set_aspect('equal', adjustable='box')
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_title('t = 0')
    # ax = fig.add_subplot(1, 2, 2)
    # l2 = ax.plot(x, U_save[Nt,:], '-', label='True')
    # ax.set_aspect('equal', adjustable='box')
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_title('t = ' + str(T))

    # plt.show()

    return U_save

# %% [markdown]
# ## Generating train data

# %%
#! 2.2 Drawing the training parameter x, to form initial condition samples
# samples = jax.random.normal(key_data_train, (num_train_samples, num_truncated_series))

a_init = jax.random.normal(key_data_train_a, (num_train_samples, 5))
b_init = jax.random.normal(key_data_train_b, (num_train_samples, 5))
x = np.linspace(0, 1, N)

U_data_train = np.zeros((num_train_samples, nt_train_data+1, N)) # Nt + 1 for including the intial data

for iii in range(num_train_samples):
    Train_u0 = np.zeros(N)
    for ii in range(5):
        Train_u0 = Train_u0 + a_init[iii,ii]*np.sin(2*np.pi*x*(ii+1)) + b_init[iii,ii]*np.sin(2*np.pi*x*(ii+1))

    U_save = generate_data_solver(Train_u0, dt, nt_train_data, N)

    U_data_train[iii,...] = U_save

print(U_data_train.shape)

#! 3 - Saving solution U
U_train_data = pd.DataFrame({'Observations': U_data_train.flatten()})
U_train_data.to_csv('data/U_wave1d_train_data_' + str(num_train_samples) + '_Nt_' + str(nt_train_data) + '.csv', index=False)


# %%
Train_data = pd.read_csv('data/U_wave1d_train_data_' + str(num_train_samples) + '_Nt_' + str(nt_train_data) + '.csv') 
Train_data = np.reshape(Train_data.to_numpy(), (num_train_samples, nt_train_data+1, N))

Train_data.shape

# %% [markdown]
# ## Generating test data

# %%
#! 2.2 Drawing the training parameter x, to form initial condition samples
b_init = jax.random.normal(key_data_test_b, (num_test_samples, 5))
a_init = jax.random.normal(key_data_test_a, (num_test_samples, 5))
x = np.linspace(0, 1, N)

U_data_test = np.zeros((num_test_samples, nt_test_data+1, N)) # Nt + 1 for including the intial data

for iii in range(num_test_samples):
    Test_u0 = np.zeros(N)
    for ii in range(5):
        Test_u0 = Test_u0 + a_init[iii,ii]*np.sin(2*np.pi*x*(ii+1)) + b_init[iii,ii]*np.sin(2*np.pi*x*(ii+1))

    U_save = generate_data_solver(Test_u0, dt_test, nt_test_data, N)

    U_data_test[iii,...] = U_save

#! 3 - Saving solution U
U_test_data = pd.DataFrame({'Observations': U_data_test.flatten()})
U_test_data.to_csv('data/U_wave1d_test_data_' + str(num_test_samples) + '_Nt_' + str(nt_test_data) + '_dt_test_' + str(dt_test) + '.csv', index=False)


# %%
Test_data = pd.read_csv('data/U_wave1d_test_data_' + str(num_test_samples) + '_Nt_' + str(nt_test_data) + '_dt_test_' + str(dt_test) + '.csv') 
Test_data = np.reshape(Test_data.to_numpy(), (num_test_samples, nt_test_data+1, N))

Test_data.shape

ns, nt, nx = U_data_train.shape
nosie_vec = jax.random.normal(key_data_noise, U_data_train.shape)
noise_level = 0.01
U_data_train_noise = np.zeros(U_data_train.shape)

for i in range(ns):
    for j in range(nt):
            U_data_train_noise[i,j,:] = U_data_train[i,j,:] + noise_level * nosie_vec[i,j,:] * np.max(U_data_train[i,j,:])

U_train_data_noise = pd.DataFrame({'Observations': U_data_train_noise.flatten()})
U_train_data_noise.to_csv('data/U_wave1d_train_data_noise_' + str(noise_level) +'_d_'  + str(num_train_samples) + '_Nt_' + str(nt_train_data) + '.csv', index=False)

print('='*20 + ' TRAIN NOISE DATA (Train x 32^2) ' + '='*20)
print(U_data_train_noise.shape)

noise_level = 0.02
U_data_train_noise = np.zeros(U_data_train.shape)

for i in range(ns):
    for j in range(nt):
            U_data_train_noise[i,j,:] = U_data_train[i,j,:] + noise_level * nosie_vec[i,j,:] * np.max(U_data_train[i,j,:])

U_train_data_noise = pd.DataFrame({'Observations': U_data_train_noise.flatten()})
U_train_data_noise.to_csv('data/U_wave1d_train_data_noise_' + str(noise_level) +'_d_'  + str(num_train_samples) + '_Nt_' + str(nt_train_data) + '.csv', index=False)

print('='*20 + ' TRAIN NOISE DATA (Train x 32^2) ' + '='*20)
print(U_data_train_noise.shape)
