#!/bin/sh
#PBS -lnodes=6:ppn=12:compute:old
#PBS -N threebody_emcee_mpi
#PBS -V

cd /data2/people/aarchiba/projects/triplesystem/code-v1
mpirun python emcee_chain_mpi.py
