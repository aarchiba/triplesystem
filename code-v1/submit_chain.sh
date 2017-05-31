#!/bin/sh
#PBS -lnodes=1:ppn=1:compute:new
#PBS -t 1-64
#PBS -N threebody_chain

cd /home/aarchiba/projects/threebody
python run_chain.py