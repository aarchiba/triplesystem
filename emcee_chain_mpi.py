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

# For some reason numpy sets the CPU affinity so we only use one processor
# Aargh! but taskset fixes it
#os.system("taskset -p 0xffffffff %d" % os.getpid())

debug = False

n_walkers = 5*84
n_steps = 100000

F = threebody.Fitter("0337+17-scott-2013-06-06",tzrmjd_middle=True)

j = 0
def lnprob(offset):
    global j
    if debug:
        print "call", j
    j = j+1
    efac = 1.3
    params = F.best_parameters.copy()
    for p,o in zip(F.parameters, offset):
        params[p] += o
    r = F.residuals(params)/F.phase_uncerts/efac
    return -0.5*np.sum(r**2)
def lnprior(offset):
    return 0

pool = emcee.utils.MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit(0)

try:
    jobid = os.environ.get('PBS_JOBID','local')
    dbdir = os.path.join('/home/aarchiba/projects/threebody/emcee-chains',jobid)
    local_dbdir = tempfile.mkdtemp()

    p0 = np.load("start-walkers.npy")
    #p0 = emcee.utils.sample_ball(
    #    np.zeros(len(F.parameters)),
    #    [F.best_errors[p] for p in F.parameters],
    #    n_walkers)

    if len(p0.shape)==2:
        sampler = emcee.EnsembleSampler(
            p0.shape[0],p0.shape[1],lnprob,
            pool=pool)
    else:
        sampler = emcee.PTSampler(
            ntemps=p0.shape[0],
            nwalkers=p0.shape[1],
            dim=p0.shape[2],
            logl=lnprob,
            logp=lnprior,
            pool=pool)

    def save():
        subprocess.check_call(['rsync','-r',
                               local_dbdir+"/",
                               "nimrod:"+dbdir+"/"])

    if debug:
        print "starting sampling loop"
    np.save(local_dbdir+"/parameters.npy", F.parameters)
    np.save(local_dbdir+"/best_parameters.npy",
            np.array([F.best_parameters[p] for p in F.parameters]))
    save()
    i = 0
    for pos, prob1, prob2 in sampler.sample(p0, iterations=n_steps,
                                            storechain=False):
        if len(p0.shape)==2:
            prob = prob1
        else:
            prob = prob2
        if debug:
            print "saving sample %d" % i
        np.save(local_dbdir+"/%06d-pos.npy" % i, pos)
        np.save(local_dbdir+"/%06d-prob.npy" % i, prob)
        save()
        i += 1
finally:
    pool.close()

