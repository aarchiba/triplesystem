#!/usr/bin/env python

import os
import subprocess
import inspect
import sys
import tempfile
import cPickle as pickle

import numpy as np
import scipy.linalg
import pymc

import minuit

import threebody
import kepler
import quad_integrate
import minuit_wrap

jobid = os.environ.get('PBS_JOBID','local')
result_file = os.path.join('/home/aarchiba/projects/threebody/minimization',jobid)
local_dir = tempfile.mkdtemp()
print result_file
print local_dir

efac = 1.67

F = threebody.Fitter()
F.mjds += efac*np.random.randn(len(F.mjds))*F.uncerts/86400.

M = minuit_wrap.Fitter(F.mfun)

for p in F.parameters:
    M.values[p] = F.best_parameters[p]
    M.errors[p] = F.best_errors[p]
M.set_normalization()

M.printMode = 3
M.tol = 100
M.eps = 1e-6
M.up = efac
M.migrad()

rf = os.path.join(local_dir,"results")
with open(rf, "wt") as f:
    pickle.dump(M.values,f)
subprocess.check_call(["rsync",rf,"nimrod:"+result_file])
