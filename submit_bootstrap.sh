#!/bin/sh
#PBS -lnodes=1:ppn=1:compute:new
#PBS -N threebody_bootstrap
#PBS -t 0-95

cd /home/aarchiba/projects/threebody
python bootstrap.py
