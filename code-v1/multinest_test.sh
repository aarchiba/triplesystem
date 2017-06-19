#!/bin/sh
#PBS -l nodes=11:ppn=8:compute
#PBS -N multinest_test
#PBS -V

cd /home/aarchiba/projects/threebody
mpirun python multinest_test.py
