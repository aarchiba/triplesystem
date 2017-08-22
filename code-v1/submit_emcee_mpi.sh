#!/bin/sh
#PBS -lnodes=6:ppn=12:compute:old
#PBS -N threebody_emcee_mpi
#PBS -V

cd /data2/people/aarchiba/projects/triplesystem/code-v1
export PATH=/users/aarchiba/.virtualenvs/triplesystem/bin:/users/aarchiba/.local/bin:/usr/local/src/presto/bin:/users/aarchiba/bin:/usr/local/src/fv5.4:/usr/local/openmpi-2.1.0/bin/:/users/aarchiba/.local/bin:/opt/rh/python27/root/usr/bin:/opt/rh/rh-git29/root/usr/bin:/opt/rh/devtoolset-6/root/usr/bin:/usr/local/src/presto/bin:/users/aarchiba/bin:/usr/local/src/fv5.4:/usr/local/openmpi-2.1.0/bin/:.:/users/aarchiba/bin:/opt/local/share/cv/bin:/usr/local/bin:/opt/local/bin:/bin:/usr/bin:/bin:/usr/etc:/usr/sbin:/sbin:/usr/local/sbin:/users/aarchiba/.local/bin:/opt/rh/python27/root/usr/bin:/opt/rh/rh-git29/root/usr/bin:/opt/rh/devtoolset-6/root/usr/bin:/usr/local/src/presto/bin:/usr/local/src/fv5.4:/usr/local/openmpi-2.1.0/bin/:/usr/lib64/qt-3.3/bin
export PYTHONPATH=


mpirun python emcee_chain_mpi.py
