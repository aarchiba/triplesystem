#!/bin/sh
#PBS -lnodes=2:ppn=8
#PBS -N threebody_optimizer

cd /home/aarchiba/projects/threebody
mpirun python run_optimizer.py
