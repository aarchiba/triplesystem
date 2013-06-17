from glob import glob
import os
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

n_walkers = 5*48
n_steps = 10000

F = threebody.Fitter()

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

pool = emcee.utils.MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit(0)

jobid = os.environ.get('PBS_JOBID','local')
dbdir = os.path.join('/home/aarchiba/projects/threebody/emcee-chains',jobid)
local_dbdir = tempfile.mkdtemp()

p0 = emcee.utils.sample_ball(
    np.zeros(len(F.parameters)),
    [F.best_errors[p] for p in F.parameters],
    n_walkers)

sampler = emcee.EnsembleSampler(
    p0.shape[0],p0.shape[1],lnprob,
    pool=pool)

def save():
    subprocess.check_call(['rsync','-r',
                           local_dbdir+"/",
                           "nimrod:"+dbdir+"/"])

if debug:
    print "starting sampling loop"
save()
i = 0
for pos, prob, state in sampler.sample(p0, iterations=n_steps):
    if debug:
        print "saving sample %d" % i
    np.save(local_dbdir+"/%06d-pos.npy" % i, pos)
    np.save(local_dbdir+"/%06d-prob.npy" % i, prob)
    #pos, prob, state = sampler.run_mcmc(p0, n_steps)
    save()
    i += 1

pool.close()

