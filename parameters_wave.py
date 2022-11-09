# sets parameters for use in Generate_data and Training scripts
import numpy as np

# Generate_data_initilizers
num_train_samples = 200
num_test_samples = 100

# ? Training inputs
num_train = 200
num_test = 100


n_seq = 1
n_seq_mc = 1

learning_rate = 1e-4
layers = 1
batch_size = 40

units = 5000

num_epochs = int(3e4)

nt_train_data = 100
nt_test_data = 500
dt = 1e-2
facdt = 1

T=.05
Nt=1000
dt = T / Nt
dt_test = dt
nt_train_data = 200
nt_test_data = 500
nu = 0.01
N = 200
dx = 1 / N

n_plot = 5
# Plot_Steps = [0, 50, 100, 200, 500]
Plot_Steps = np.linspace(0,nt_test_data, n_plot)
