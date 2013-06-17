#!/bin/sh
#PBS -lnodes=1:ppn=12:compute:new
#PBS -N threebody_emcee

cd /home/aarchiba/projects/threebody
python emcee_chain.py