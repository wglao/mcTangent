import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import wandb

from matplotlib import cm  # Colour map
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import jax
from jax.nn.initializers import normal, zeros
from jax import value_and_grad, vmap, random, jit, lax
import jax.numpy as jnp
from jax.example_libraries import stax, optimizers

import time
import pickle

# from jax.config import config
# config.update("jax_enable_x64", True)

#! Step : 0 - Generate_data_initilizers

mc_alpha = 1e5

# initialize physic parameters
# initialize parameters
import parameters_wave
from parameters_wave import *

# ? Step 0.2 - Uploading wandb
filename = 'wave1d_dt_train-test_' + str(dt) + '-' + str(dt_test) + '_seq_n_mc_' + str(n_seq_mc) +'_forward_mc_train_d' + str(num_train) + '_alpha_' + str(mc_alpha) + '_lr_' + str(learning_rate) + '_batch_' + str(batch_size) + '_nseq_' + str(n_seq) + '_layer_' + str(layers) + 'neurons' + str(units) + '_epochs_' + str(num_epochs)

wandb.init(project="mcTangent")
wandb.config.problem = 'wave1D'
wandb.config.mc_alpha = mc_alpha
wandb.config.learning_rate = learning_rate
wandb.config.num_epochs = num_epochs
wandb.config.batch_size = batch_size
wandb.config.n_seq = n_seq
wandb.config.layer = layers
wandb.config.method = 'Dense_net'

# ? Step 0.3 - Spectral method for 2D Navier-Stoke equation initialize parameters

#! Step 1: Loading data
# ? 1.1 Loading data by pandas
print('=' * 20 + ' >>')
print('Loading train data ...')

Train_data = pd.read_csv('data/U_wave1d_train_data_' + str(num_train) + '_Nt_' + str(nt_train_data) + '.csv')
Train_data = np.reshape(Train_data.to_numpy(), (num_train, nt_train_data+1, N))

print(Train_data.shape)
print('=' * 20 + ' >>')
print('Loading test data ...')

Test_data = pd.read_csv('data/U_wave1d_test_data_' + str(num_test) + '_Nt_' + str(nt_test_data) + '_dt_test_' + str(dt_test) + '.csv')
Test_data = np.reshape(Test_data.to_numpy(), (num_test, nt_test_data+1, N))

print(Test_data.shape)


#! Step 2: Building up a neural network
forward_pass_int, _ = stax.serial(
    stax.Dense(units, W_init=normal(0.02), b_init=zeros), stax.Relu,
    stax.Dense(N, W_init=normal(0.02), b_init=zeros),
)
_, init_params = forward_pass_int(random.PRNGKey(0), (N,))

W1, b1 = init_params[0]
W2, b2 = init_params[-1]

def ReLU(x):
    """ Rectified Linear Unit (ReLU) activation function """
    return jnp.maximum(0, x)

def Dense(inputs, W, b):
    return jnp.dot(inputs, W) + b

def forward_pass(params, u):
    W1, W2, b1, b2 = params
    u = Dense(ReLU(Dense(u, W1, b1)), W2, b2)
    return u

init_params = [W1, W2, b1, b2]

print('=' * 20 + ' >> Success!')


#! Step 3: Forward solver (single time step)
def single_solve_forward(un):
    # use different difference schemes for edge case
    lu = len(un)
    u = un + (- dt / dx * (jnp.roll(un, -1) - jnp.roll(un, 1)) / 2 )
    uleft = un[0] + (dt / dx / 2 * (3*un[0] - 4*un[1] + un[2]))
    uright = un[lu-1] + (dt / -dx / 2 * (3*un[lu-1] - 4*un[lu-2] + un[lu-3]))
    u = u.at[0].set(uleft)
    u = u.at[lu-1].set(uright)
    return u

#@jit
def single_forward_pass(params, un):
    u = un - facdt * dt * forward_pass(params, un)
    return u.flatten()


#! Step 4: Loss functions and relative error/accuracy rate function
# ? 4.1 For one time step data (1, 1, Nx)
def MSE(pred, true):
    return jnp.mean(jnp.square(pred - true))

def squential_mc(i, args):
    
    loss_mc, u_mc, u_ml, params = args
    u_ml_next = single_forward_pass(params, u_ml)
    u_mc_next = single_solve_forward(u_mc)
    
    loss_mc += MSE(u_mc, u_ml_next)

    return loss_mc, u_mc_next, u_ml_next, params

def squential_ml_second_phase(i, args):
    ''' I have checked this loss function!'''

    loss_ml, loss_mc, u_ml, u_true, params = args
    
    # This is u_mc for the current
    u_mc = single_solve_forward(u_ml)
    
    # This is u_ml for the next step
    u_ml_next = single_forward_pass(params, u_ml)
    
    # # The model-constrained loss 
    # loss_mc += MSE(u_mc, u_true[i+1,:]) 

    # The forward model-constrained loss
    # loss_mc, _, _, _ = lax.fori_loop(0, n_seq_mc, squential_mc, (loss_mc, u_mc, u_ml, params))
    loss_mc += MSE(u_mc, u_ml_next)
    
    # The machine learning term loss
    loss_ml += MSE(u_ml, u_true[i,:])

    return loss_ml, loss_mc, u_ml_next, u_true, params


def loss_one_sample_one_time(params, u):
    loss_ml = 0
    loss_mc = 0

    # first step prediction
    u_ml = single_forward_pass(params, u[0, :])

    # for the following steps up to sequential steps n_seq
    loss_ml,loss_mc, u_ml, _, _ = lax.fori_loop(1, n_seq+1, squential_ml_second_phase, (loss_ml, loss_mc, u_ml, u, params))
    loss_ml += MSE(u_ml, u[-1, :])

    return loss_ml + mc_alpha * loss_mc

loss_one_sample_one_time_batch = vmap(loss_one_sample_one_time, in_axes=(None, 0), out_axes=0)

# ? 4.2 For one sample of (1, Nt, Nx)
#@jit
def loss_one_sample(params, u_one_sample):
    return jnp.sum(loss_one_sample_one_time_batch(params, u_one_sample))

loss_one_sample_batch = vmap(loss_one_sample, in_axes=(None, 0), out_axes=0)

# ? 4.3 For the whole data (n_samples, Nt, Nx)
# ? This step transform data to disired shape for training (n_train_samples, Nt, Nx) -> (n_train_samples, Nt, n_seq, Nx)
#@jit
def transform_one_sample_data(u_one_sample):
    u_out = jnp.zeros((nt_train_data - n_seq - 1, n_seq+2, N))
    for i in range(nt_train_data-n_seq-1):
        u_out = u_out.at[i, :, :].set(u_one_sample[i:i + n_seq + 2, :])
    return u_out

transform_one_sample_data_batch = vmap(transform_one_sample_data, in_axes=0)

#@jit
def LossmcDNN(params, data):
    return jnp.sum(loss_one_sample_batch(params, transform_one_sample_data_batch(data)))


#! Step 5: Computing test error, predictions over all time steps
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


@jit
def test_acc(params, Test_set):
    return MSE(neural_solver_batch(params, Test_set), Test_set)

#! Step 6: Epoch loops fucntions and training settings
def body_fun(i, args):
    loss, opt_state, data = args

    data_batch = lax.dynamic_slice_in_dim(data, i * batch_size, batch_size)

    loss, gradients = value_and_grad(LossmcDNN)(
        opt_get_params(opt_state), data_batch)

    opt_state = opt_update(i, gradients, opt_state)

    return loss/batch_size, opt_state, data


@jit
def run_epoch(opt_state, data):
    loss = 0
    return lax.fori_loop(0, num_batches, body_fun, (loss, opt_state, data))


def TrainModel(train_data, test_data, num_epochs, opt_state):

    test_accuracy_min = 100
    epoch_min = 1

    for epoch in range(1, num_epochs+1):
        
        t1 = time.time()
        train_loss, opt_state, _ = run_epoch(opt_state, train_data)
        t2 = time.time()

        test_accuracy = test_acc(opt_get_params(opt_state), test_data)

        if test_accuracy_min >= test_accuracy:
            test_accuracy_min = test_accuracy
            epoch_min = epoch
            optimal_opt_state = opt_state

        if epoch % 1000 == 0:  # Print MSE every 1000 epochs
            print("Data_d {:d} n_seq {:d} batch {:d} time {:.2e}s loss {:.2e} TE {:.2e}  TE_min {:.2e} EPmin {:d} EP {} ".format(
                num_train, n_seq, batch_size, t2 - t1, train_loss, test_accuracy, test_accuracy_min, epoch_min, epoch))

        wandb.log({"Train loss": float(train_loss), "Test Error": float(test_accuracy), 'TEST MIN': float(test_accuracy_min), 'Epoch' : float(epoch)})

    return optimal_opt_state, opt_state


num_complete_batches, leftover = divmod(num_train, batch_size)
num_batches = num_complete_batches + bool(leftover)

opt_int, opt_update, opt_get_params = optimizers.adam(learning_rate)
opt_state = opt_int(init_params)

best_opt_state, end_opt_state = TrainModel(Train_data, Test_data, num_epochs, opt_state)

optimum_params = opt_get_params(best_opt_state)
End_params = opt_get_params(end_opt_state)
# from jax.example_libraries.optimizers import optimizers

trained_params = optimizers.unpack_optimizer_state(end_opt_state)
pickle.dump(trained_params, open('Network/End_' + filename, "wb"))

trained_params = optimizers.unpack_optimizer_state(best_opt_state)
pickle.dump(trained_params, open('Network/Best_' + filename, "wb"))


# %% Plot predictions
U_pred = neural_solver_batch(optimum_params, Test_data)[0, :, :]
U_true = Test_data[0, :, :]

x = np.linspace(0, 1, N)


def plot_compare(U_True, U_Pred, filename):

    fig = plt.figure(figsize=(32,10))
    fig.patch.set_facecolor('xkcd:white')

    # Compare solutions
    for i in range(5):
        ut = jnp.reshape(U_True[Plot_Steps[i], :], (N, 1))
        up = jnp.reshape(U_Pred[Plot_Steps[i], :], (N, 1))
        ax = fig.add_subplot(1, 5, i+1)
        l1 = ax.plot(x, ut, '-', linewidth=2, label='True')
        l2 = ax.plot(x, up, '--', linewidth=2, label='Predicted')
        ax.set_aspect('auto', adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('t = ' + str(Plot_Steps[i]))

        if i == 1:
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center')

    plt.savefig('figs/' + filename + '.png', bbox_inches='tight')


plot_compare(U_true, U_pred, filename)
