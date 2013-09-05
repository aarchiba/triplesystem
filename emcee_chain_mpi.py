from glob import glob
import os
import sys
import tempfile
import subprocess
import cPickle as pickle
import threading
import time

import numpy as np
import scipy.linalg

import emcee

import threebody

# For some reason numpy sets the CPU affinity so we only use one processor
# Aargh! but taskset fixes it
# Not needed if using MPI
#os.system("taskset -p 0xffffffff %d" % os.getpid())

debug = False
trust_nfs = True

n_steps = 100000

#fitter_params = dict(files="0337+17-scott-2013-06-06",
#                     tzrmjd_middle='weighted',
#                     priors=['dbeta','dgamma'],
#                     fit_pos=True, parfile="0337_tempo2_nobinary.par",
#                     ppn_mode='heavysimple')
fitter_params = dict(files="0337+17-scott-2013-08-29",
                     tzrmjd_middle='auto',
                     fit_pos=True, fit_pm=True, fit_px=True,
                     t2_astrometry=True,
                     parfile="0337_tempo2_nobinary.par",
                     ppn_mode='GR')
F = threebody.Fitter(**fitter_params)
#F = threebody.Fitter("0337+17-scott-2013-06-06",
#                     tzrmjd_middle='weighted',
#                     ppn_mode='GR')
#F = threebody.Fitter("0337+17-scott-2013-06-06",
#                     tzrmjd_middle='weighted',
#                     ppn_mode='heavysimple')
#F = threebody.Fitter("0337+17-scott-2013-06-06",
#                     tzrmjd_middle='weighted',
#                     priors=['dbeta','dgamma'],
#                     ppn_mode='heavysimple')
# F = threebody.Fitter("0337+17-scott-2013-06-06",
#                      tzrmjd_middle='weighted',
#                      priors=['dbeta','dgamma'],
#                      use_quad=True, tol=1e-20,
#                      ppn_mode='heavysimple')
#F = threebody.Fitter("0337+17-scott-2013-06-06",
#                     tzrmjd_middle='weighted',
#                     only_tels=['AO1440','AO1350','GBT1500'],
#                     ppn_mode='heavysimple')
# F = threebody.Fitter("0337+17-scott-2013-06-06",
#                      tzrmjd_middle='weighted',
#                      only_tels=['WSRT1400'],
#                      priors=['dbeta','dgamma'],
#                      ppn_mode='heavysimple')
# F = threebody.Fitter("0337+17-scott-2013-06-06",
#                      tzrmjd_middle='weighted',
#                      only_tels=['GBT1500'],
#                      priors=['dbeta','dgamma'],
#                      ppn_mode='heavysimple')
# F = threebody.Fitter("0337+17-scott-2013-06-06",
#                      tzrmjd_middle='weighted',
#                      only_tels=['AO1440','AO1350'],
#                      priors=['dbeta','dgamma'],
#                      ppn_mode='heavysimple')
#F = threebody.Fitter("0337+17-scott-2013-06-06",
#                     tzrmjd_middle='weighted',
#                     priors=['dbeta','dgamma'],
#                     fit_pos=True, parfile="0337_tempo2_nobinary.par",
#                     ppn_mode='heavysimple')


j = 0
def lnprob(offset):
    global j
    if debug:
        print "call", j
    j = j+1
    efac = 1.3
    params = F.best_parameters.copy()
    if len(offset)!=len(F.parameters):
        raise ValueError("Parameter mismatch")
    for p,o in zip(F.parameters, offset):
        params[p] += o
    return F.lnprob(params)
def lnprior(offset):
    params = F.best_parameters.copy()
    for p,o in zip(F.parameters, offset):
        params[p] += o
    return F.lnprior(params)

pool = emcee.utils.MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit(0)

try:
    jobid = os.environ.get('PBS_JOBID','local')
    dbdir = os.path.join('/home/aarchiba/projects/threebody/emcee-chains',jobid)
    if trust_nfs:
        local_dbdir = dbdir
        os.mkdir(dbdir)
    else:
        local_dbdir = tempfile.mkdtemp()

    p0 = np.load("start-walkers.npy")

    if len(p0.shape)==2:
        sampler = emcee.EnsembleSampler(
            p0.shape[0],p0.shape[1],
            lambda offset: lnprob(offset)+lnprior(offset),
            pool=pool) # FIXME: untested
    else:
        sampler = emcee.PTSampler(
            ntemps=p0.shape[0],
            nwalkers=p0.shape[1],
            dim=p0.shape[2],
            logl=lnprob,
            logp=lnprior,
            pool=pool)

    def save():
        subprocess.check_call(['rsync','-rt', '--append-verify',
                               local_dbdir+"/",
                               "nimrod:"+dbdir+"/"])
    if not trust_nfs:
        # Run saving in the background so it doesn't interfere with computation
        done = False
        def save_loop():
            if debug:
                print "starting saving loop"
            while not done:
                save()
                time.sleep(30)
                if debug:
                    print "saved at", time.asctime()
        the_thread = threading.Thread(target=save_loop)
        the_thread.start()
    if debug:
        print "starting sampling loop"
    with open(local_dbdir+"/fitter_params.pickle","wb") as f:
        pickle.dump(fitter_params,f)
    np.save(local_dbdir+"/parameters.npy", F.parameters)
    np.save(local_dbdir+"/best_parameters.npy",
            np.array([F.best_parameters[p] for p in F.parameters]))
    i = 0
    for pos, prob1, prob2 in sampler.sample(p0, iterations=n_steps,
                                            storechain=False):
        if len(p0.shape)==2:
            prob = prob1
        else:
            prob = prob2
        if debug:
            print "writing sample %d" % i
        np.save(local_dbdir+"/%06d-pos.npy" % i, pos)
        np.save(local_dbdir+"/%06d-prob.npy" % i, prob)
        i += 1
finally:
    if not trust_nfs:
        done = True
    pool.close()

