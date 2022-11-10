from jax import vmap, jit
import jax.numpy as jnp
from jax.example_libraries import stax
from parameters_wave import *

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