from __future__ import division
import os, shutil
import logging
import pickle
import numpy as np
import scipy.stats

import pymultinest

import threebody

jobid = os.environ.get('PBS_JOBID','local')
dbdir = os.path.join('/home/aarchiba/projects/threebody/multinest-chains',jobid)
try:
    os.mkdir(dbdir)
except OSError:
    pass
logger = logging.getLogger()
outbase = os.path.join(dbdir,"chain")

if True:
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(dbdir,"rank-%s.log" % os.environ['OMPI_COMM_WORLD_RANK']))
    formatter = logging.Formatter('%(asctime)s - %(module)s:%(funcName)s:%(lineno)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

logger.debug("creating Fitter")

fitter_params = pickle.load(open("multinest_params.pickle","rb"))
shutil.copy("multinest_params.pickle", dbdir)

F = threebody.Fitter(**fitter_params)
logger.debug("Fitter created")

j = 0
def lnprob(cube, ndim, nparams):
    global j
    logger.debug("call %d" % j)
    j = j+1
    efac = 1.3
    params = F.best_parameters.copy()
    for i,p in enumerate(F.parameters):
        params[p] = cube[i]
    logger.debug("started lnprob computation")
    r = F.lnprob(params)
    logger.debug("finished lnprob computation with %s" % r)
    return r

def prior(cube, ndim, nparams):
    for i,p in enumerate(F.parameters):
        t, v = threebody.multinest_prior[p]
        if t=='range':
            cube[i] = cube[i]*(v[1]-v[0])+v[0]
        elif t=='normal':
            cube[i] = scipy.stats.norm.isf(cube[i])*v[1]+v[0]
        else:
            raise ValueError("Prior not understood")

logger.debug("Starting multinest")
pymultinest.run(lnprob, prior, len(F.parameters),
                outputfiles_basename=outbase,
                init_MPI=False,
                resume=True, verbose=True)
