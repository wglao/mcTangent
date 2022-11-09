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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm  # Colour map
import matplotlib.animation as animation

# %%
# ## 0 - Generate_data_initilizers
num_truncated_series = 15
num_train_samples = 200
num_test_samples = 10

# random seed for training data
key_data_train = random.PRNGKey(1)

# random seed for test data
key_data_test = random.PRNGKey(2)

key_data_noise = random.PRNGKey(3)

#! 1 - Spectral method for 2D Navier-Stoke equation initialize parameters
# initialize parameters
from parameters import *

shift_top_right = np.eye(N,N,1)
shift_top_right[-1,0] = 1
shift_bot_left = np.eye(N,N,-1)
shift_bot_left[0,-1] = 1

shift_top_right = jnp.asarray(shift_top_right)
shift_bot_left = jnp.asarray(shift_bot_left)

x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
X, Y = np.meshgrid(x, y)

# ## 2 - Draw initial condition
#! 2.1 Saving the eigenvector and eigenvalues
df_Basis = pd.read_csv('data/Basis_Modes.csv')
Modes_100 = np.reshape(df_Basis.to_numpy(), (N**2, 100))

Basis = Modes_100[:, :num_truncated_series]

Solution_samples_array = pd.DataFrame({'Basis': Basis.flatten()})
Solution_samples_array.to_csv('data/Basis.csv', index=False)

# %%
def single_step_forward_solver(un, vn, dt):
    u = (
        un
        - dt / dx * jnp.multiply(un , (jnp.dot(un , shift_bot_left) - jnp.dot(un , shift_top_right))) / 2
        - dt / dy * jnp.multiply(un , (jnp.dot(shift_top_right , vn) - jnp.dot(shift_bot_left , vn))) / 2
        + nu * dt / dx ** 2 * (jnp.dot(un , shift_bot_left) - 2 * un + jnp.dot(un , shift_top_right))
        + nu * dt / dy ** 2 * (jnp.dot(shift_top_right , un) - 2 * un + jnp.dot(shift_bot_left , un))
    )
    
    v = (
        vn
        -  dt / dx * jnp.multiply(vn , jnp.dot(un , shift_bot_left) - jnp.dot(un , shift_top_right)) / 2
        - dt / dy * jnp.multiply(vn , jnp.dot(shift_top_right , vn) - jnp.dot(shift_bot_left , vn)) / 2
        + nu * dt / dx ** 2 * (jnp.dot(vn , shift_bot_left) - 2 * vn + jnp.dot(vn , shift_top_right))
        + nu * dt / dy ** 2 * (jnp.dot(shift_top_right , vn) - 2 * vn + jnp.dot(shift_bot_left , vn))
    )
    return u, vn

def generate_data_solver(IBC, dt, Nt):
    U_save = np.zeros((Nt+1, N**2)) # Nt + 1 for including the intial data
    V_save = np.zeros((Nt+1, N**2))

    u = jnp.reshape(IBC, (N,N))
    v = jnp.ones(u.shape)

    U_save[0, :] = u.flatten()
    V_save[0, :] = v.flatten()

    for i in  range(1, Nt+1):
        u, v = single_step_forward_solver(u,v, dt)
        U_save[i, :] = u.flatten()
        V_save[i, :] = v.flatten()

    return U_save, V_save

# %% [markdown]
# ## Generating train data

# %%
#! 2.2 Drawing the training parameter x, to form initial condition samples
samples = jax.random.normal(key_data_train, (num_train_samples, num_truncated_series))
Train_u0 = jnp.exp(jnp.asarray(samples @ Basis.T))

U_data_train = np.zeros((num_train_samples, nt_train_data+1, N**2)) # Nt + 1 for including the intial data
V_data_train = np.zeros((num_train_samples, nt_train_data+1, N**2))

for iii in range(num_train_samples):
    U_save, V_save = generate_data_solver(Train_u0[iii,:], dt, nt_train_data)

    U_data_train[iii,...] = U_save
    V_data_train[iii,...] = V_save

print(U_data_train.shape)

#! 3 - Saving solution U
U_train_data = pd.DataFrame({'Observations': U_data_train.flatten()})
U_train_data.to_csv('data/U_burger_train_data' + str(num_train_samples) + '_Nt_' + str(nt_train_data) + '.csv', index=False)


# %%
Train_data = pd.read_csv('data/U_burger_train_data' + str(num_train_samples) + '_Nt_' + str(nt_train_data) + '.csv') 
Train_data = np.reshape(Train_data.to_numpy(), (num_train_samples, nt_train_data+1, N**2))

Train_data.shape

# %% [markdown]
# ## Generating test data

# %%
#! 2.2 Drawing the training parameter x, to form initial condition samples
samples = jax.random.normal(key_data_test, (num_test_samples, num_truncated_series))
Test_u0 = jnp.exp(jnp.asarray(samples @ Basis.T))

U_data_test = np.zeros((num_test_samples, nt_test_data+1, N**2)) # Nt + 1 for including the intial data
V_data_test = np.zeros((num_test_samples, nt_test_data+1, N**2))

for iii in range(num_test_samples):
    U_save, V_save = generate_data_solver(Test_u0[iii,:], dt_test, nt_test_data)

    U_data_test[iii,...] = U_save
    V_data_test[iii,...] = V_save

#! 3 - Saving solution U
U_test_data = pd.DataFrame({'Observations': U_data_test.flatten()})
U_test_data.to_csv('data/U_burger_test_data' + str(num_test_samples) + '_Nt_' + str(nt_test_data) + '_dt_test_' + str(dt_test) + '.csv', index=False)


# %%
Test_data = pd.read_csv('data/U_burger_test_data' + str(num_test_samples) + '_Nt_' + str(nt_test_data) + '_dt_test_' + str(dt_test) + '.csv') 
Test_data = np.reshape(Test_data.to_numpy(), (num_test_samples, nt_test_data+1, N**2))

Test_data.shape

ns, nt, nx = U_data_train.shape
nosie_vec = jax.random.normal(key_data_noise, U_data_train.shape)
noise_level = 0.01
U_data_train_noise = np.zeros(U_data_train.shape)

for i in range(ns):
    for j in range(nt):
            U_data_train_noise[i,j,:] = U_data_train[i,j,:] + noise_level * nosie_vec[i,j,:] * np.max(U_data_train[i,j,:])

U_train_data_noise = pd.DataFrame({'Observations': U_data_train_noise.flatten()})
U_train_data_noise.to_csv('data/U_burger_train_data_noise_' + str(noise_level) +'_d_'  + str(num_train_samples) + '_Nt_' + str(nt_train_data) + '.csv', index=False)

print('='*20 + ' TRAIN NOISE DATA (Train x 32^2) ' + '='*20)
print(U_data_train_noise.shape)

noise_level = 0.02
U_data_train_noise = np.zeros(U_data_train.shape)

for i in range(ns):
    for j in range(nt):
            U_data_train_noise[i,j,:] = U_data_train[i,j,:] + noise_level * nosie_vec[i,j,:] * np.max(U_data_train[i,j,:])

U_train_data_noise = pd.DataFrame({'Observations': U_data_train_noise.flatten()})
U_train_data_noise.to_csv('data/U_burger_train_data_noise_' + str(noise_level) +'_d_'  + str(num_train_samples) + '_Nt_' + str(nt_train_data) + '.csv', index=False)

print('='*20 + ' TRAIN NOISE DATA (Train x 32^2) ' + '='*20)
print(U_data_train_noise.shape)
