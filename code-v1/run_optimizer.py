from glob import glob
import os
import sys
import tempfile
import subprocess
import cPickle as pickle

import numpy as np
import scipy.linalg

import emcee

import threebody
import optimizer

# For some reason numpy sets the CPU affinity so we only use one processor
# Aargh! but taskset fixes it
#os.system("taskset -p 0xffffffff %d" % os.getpid())

debug = False

F = threebody.Fitter("0337+17-scott-2013-06-06",tzrmjd_middle=True)

def chi2(offset):
    if np.shape(offset) != (len(F.parameters),):
        raise ValueError("Received an offset of shape %s" % (np.shape(offset),))
    params = F.best_parameters.copy()
    for p,o in zip(F.parameters, offset):
        params[p] += o
    r = F.residuals(params)/F.phase_uncerts
    return np.sum(r**2)

pool = emcee.utils.MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit(0)

try:
    jobid = os.environ.get('PBS_JOBID','local')
    dbdir = os.path.join('/home/aarchiba/projects/threebody/optimizer-runs',jobid)
    local_dbdir = tempfile.mkdtemp()

    p0 = np.load("start-optimizer.npy")
    #p0 = emcee.utils.sample_ball(
    #    np.zeros(len(F.parameters)),
    #    [F.best_errors[p] for p in F.parameters],
    #    n_walkers)

    p0 = list(p0.reshape((-1,p0.shape[-1])))

    def save():
        subprocess.check_call(['rsync','-r',
                               local_dbdir+"/",
                               "nimrod:"+dbdir+"/"])

    np.save(local_dbdir+"/parameters.npy", F.parameters)
    np.save(local_dbdir+"/best_parameters.npy",
            np.array([F.best_parameters[p] for p in F.parameters]))
    save()

    i = 0
    for (best_pos, best_chi2,
         ps, chi2s) in optimizer.optimize(p0, chi2,
                                          pool=pool,
                                          explore=2,
                                          adapt_covariance=False):
        np.save(local_dbdir+"/best_pos.npy", best_pos)
        np.save(local_dbdir+"/best_chi2.npy", best_chi2)
        np.save(local_dbdir+"/%06d-ps.npy" % i, ps)
        np.save(local_dbdir+"/%06d-chi2s.npy" % i, chi2s)
        save()
        i += 1
finally:
    pool.close()

