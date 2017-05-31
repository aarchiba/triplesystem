#!/usr/bin/env python

import os
import subprocess
import inspect
import sys
import tempfile

import numpy as np
import scipy.linalg
import pymc

import minuit

import threebody
import kepler
import quad_integrate
import minuit_wrap

jobid = os.environ['PBS_JOBID']
dbdir = os.path.join('/home/aarchiba/projects/threebody/chains',jobid)
local_dbdir = tempfile.mkdtemp()

F = threebody.Fitter()

# Any distribution I put here is a prior, so let's be broad
#spread = 10000.
#offset = pymc.Normal('offset',
#    mu=np.zeros(len(F.parameters)),
#    tau=spread**(-2)*np.array([F.best_errors[p] for p in F.parameters])**(-2))

# Flat priors! This is not kosher but should be fine
@pymc.stochastic()
def offset(value=np.zeros(len(F.parameters))):
    return 0

@pymc.deterministic(trace=False)
def resids(offset=offset):
    params = F.best_parameters.copy()
    for p,o in zip(F.parameters, offset):
        params[p] += o
    return F.residuals(params)/F.phase_uncerts
efac = 1.3
result = pymc.Normal('result',
                     mu=resids,
                     tau=efac**(-2.),
                     value=np.zeros(len(F.mjds)),
                     trace=False,
                     observed=True)
 
fudge_step = 5e-2 # to get better acceptance at first
M = pymc.MCMC([offset,resids,result],
              db='txt', dbname=local_dbdir)
#M.use_step_method(pymc.AdaptiveMetropolis, offset,
#                  cov=np.diag([fudge_step*F.best_errors[p]**2 for p in F.parameters]))
# Retunes every 1000 or so
M.use_step_method(pymc.Metropolis, offset,
    proposal_sd=np.array([fudge_step*F.best_errors[p] for p in F.parameters]),
    proposal_distribution='Normal')
M.sample(burn=0,iter=30000,tune_throughout=False)
M.db.close()
subprocess.check_call(['rsync','-r','--append',
                       local_dbdir+"/",
                       "nimrod:"+dbdir+"/"])

