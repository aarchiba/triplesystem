from glob import glob
import os
import tempfile
import cPickle as pickle

import numpy as np
import scipy.linalg

import emcee

import threebody

# For some reason numpy sets the CPU affinity so we only use one processor
# Aargh! but taskset fixes it
os.system("taskset -p 0xffffffff %d" % os.getpid())

n_walkers = 250
n_steps = 100
n_cores = 12

def lnprob(offset, F):
    efac = 1.3
    params = F.best_parameters.copy()
    for p,o in zip(F.parameters, offset):
        params[p] += o
    r = F.residuals(params)/F.phase_uncerts/efac
    return -0.5*np.sum(r**2)

if __name__=='__main__':
    jobid = os.environ.get('PBS_JOBID','local')
    dbdir = os.path.join('/home/aarchiba/projects/threebody/emcee-chains',jobid)
    local_dbdir = tempfile.mkdtemp()

    F = threebody.Fitter()

    p0 = emcee.utils.sample_ball(
        np.zeros(len(F.parameters)),
        [F.best_errors[p] for p in F.parameters],
        n_walkers)

    sampler = emcee.EnsembleSampler(
        p0.shape[0],p0.shape[1],lnprob,
        args=(F,),
        threads=n_cores)

    pos, prob, state = sampler.run_mcmc(p0, n_steps)

    pickle.save(sampler,os.path.join(local_dbdir,"sampler.pickle"))
    subprocess.check_call(['rsync','-r','--append',
                           local_dbdir+"/",
                           "nimrod:"+dbdir+"/"])
