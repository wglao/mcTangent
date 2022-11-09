#!/bin/bash
#----------------------------------------------------
# Slurm job script
#   for TACC Lonestar6 GPU
#
#   *** GPU Job in gpu-a100 Queue***
# 
# Last revised: November 6, 2022
#
# Notes:
#
#  -- Copy/edit this script as desired.  Launch by executing
#     "sbatch train.slurm" on a Lonestar6 login node.
#
#  -- GPU codes run on 4 nodes (upper case N = 4).
#
#  -- Use TACC's launcher utility to run multiple serial 
#       executables at the same time, execute "module load launcher" 
#       followed by "module help launcher".
#----------------------------------------------------

#SBATCH -J mcTtrain           # Job name
#SBATCH -o mcTtrain.o%j       # Name of stdout output file
#SBATCH -e mcTtrain.e%j       # Name of stderr error file
#SBATCH -p gpu-a100           # Queue (partition) name
#SBATCH -N 1                  # Total # of nodes
#SBATCH -n 128                  # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 01:00:00           # Run time (hh:mm:ss)
#SBATCH --mail-type=all       # Send email at begin and end of job
#SBATCH -A DMS21058           # Project/Allocation name (req'd if you have more than 1)
#SBATCH --mail-user=wgl@tacc.utexas.edu

# Any other commands must follow all #SBATCH directives...
cp forward_wave.py Generate_data_wave.py parameters_wave.py $SCRATCH
cd $SCRATCH
mkdir data; mkdir figs; mkdir Network

module list
pwd
date

# Launch serial code...
python Generate_data_wave.py
python forward_wave.py

cd $HOME/PHO-ICES/mcTangent
cp $SCRATCH/* 
rm -rf $SCRATCH/*

