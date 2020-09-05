#!/bin/bash
#SBATCH --job-name=squash
#SBATCH --ntasks=1                          # Run a single task   
#SBATCH --cpus-per-task=1                    # Number of CPU cores per task
# #SBATCH --nodes=8
#SBATCH --time=11:11:11
#SBATCH --output=squash_%j.log          # Standard output and error log
#SBATCH --account=phys-bout-2019             # Project account
#SBATCH --mem=200gb

# # echo "Running mpi_example on $SLURM_CPUS_ON_NODE CPU cores"
# export HDF5_USE_FILE_LOCKING=FALSE
# # module unload numlib/PETSc/3.9.3-foss-2018b

# export LD_LIBRARY_PATH=/mnt/lustre/groups/phys-bout-2019/install/lib64/:$LD_LIBRARY_PATH
# export PETSC_DIR=/users/hm1234/scratch/next/petsc2/petsc:$PETSC_DIR
# export LD_LIBRARY_PATH=/opt/apps/easybuild/software/compiler/GCCcore/7.3.0/lib64/:$LD_LIBRARY_PATH
# # export PETSC_DIR=/users/hm1234/scratch/next/petsc/petsc-3.11.1:$PETSC_DIR
# # export LD_LIBRARY_PATH=/users/hm1234/scratch/next/petsc/petsc-3.11.1/lib:$LD_LIBRARY_PATH

echo trying

# cd /users/hm1234/scratch/test2/impurity-scan/test

# /mnt/lustre/groups/phys-bout-2019/hermes-2/hermes-2
# /mnt/lustre/groups/phys-bout-2019/hermes-2-next/hermes-2
# /users/hm1234/scratch/hermes-2/hermes-2

# solver:type=pvode

export OMP_NUM_THREADS=1

python ~/scratch/hermes-sim-tools/x-analysis.py

echo exited
