#!/bin/sh
#PBS -lnodes=4:ppn=12:compute:new
#PBS -N threebody_emcee_mpi

cd /home/aarchiba/projects/threebody
mpirun python emcee_chain_mpi.py