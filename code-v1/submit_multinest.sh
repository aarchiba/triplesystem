#!/bin/sh
#PBS -l nodes=11:ppn=8:compute
#PBS -N multinest_run
#PBS -V

cd /home/aarchiba/projects/threebody
mpirun python multinest_run.py
