#!/bin/bash
#----------------------------------------------------
# Slurm job script
#   for TACC Lonestar6 GPU
#
#   *** Test job in development Queue***
# 
# Last revised: November 6, 2022
#
# Notes:
#
#  -- Copy/edit this script as desired.  Launch by executing
#     "sbatch train.slurm" on a Lonestar6 login node.
#
#  -- development queue codes run on 4 nodes (upper case N = 4).
#
#  -- Use TACC's launcher utility to run multiple serial 
#       executables at the same time, execute "module load launcher" 
#       followed by "module help launcher".
#----------------------------------------------------

#SBATCH -J mcTtrain           # Job name
#SBATCH -o mcTtrain.o%j       # Name of stdout output file
#SBATCH -e mcTtrain.e%j       # Name of stderr error file
#SBATCH -p development           # Queue (partition) name
#SBATCH -N 4                  # Total # of nodes
#SBATCH -n 512                  # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 01:30:00           # Run time (hh:mm:ss)
#SBATCH --mail-type=all       # Send email at begin and end of job
#SBATCH -A DMS21058           # Project/Allocation name (req'd if you have more than 1)
#SBATCH --mail-user=wgl@tacc.utexas.edu

# Any other commands must follow all #SBATCH directives...
cp -r toMove $SCRATCH
cd $SCRATCH/toMove

module list
pwd
date

# Launch serial code...
./forward_wave.py         # Do not use ibrun or any other MPI launcher

cp $SCRATCH/toMove/figs/* ./figs
rm -rf $SCRATCH/toMove

cd $HOME/PHO-ICES/mcTangent