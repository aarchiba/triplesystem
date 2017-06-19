from glob import glob
import os
import sys
import tempfile
import subprocess
import cPickle as pickle
import threading
import time
import logging

import numpy as np
import scipy.linalg

import emcee

import threebody

jobid = os.environ.get('PBS_JOBID','local')
dbdir = os.path.join('/home/aarchiba/projects/threebody/emcee-chains',jobid)
try:
    os.mkdir(dbdir)
except OSError:
    pass
logger = logging.getLogger()

if True:
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(
        dbdir,"rank-%s.log" % os.environ['OMPI_COMM_WORLD_RANK']))
    formatter = logging.Formatter(
        '%(asctime)s - %(module)s:%(funcName)s:%(lineno)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

# For some reason numpy sets the CPU affinity so we only use one processor
# Aargh! but taskset fixes it
# Not needed if using MPI
#os.system("taskset -p 0xffffffff %d" % os.getpid())

trust_nfs = True

n_steps = 100000

logger.debug("creating Fitter")
#mode = 'sep-2014-01'
mode = 'file:emcee_params.pickle'
#only_tels = ('AO1350','AO1440')
#only_tels = ('GBT1500',)
#only_tels = ('WSRT1400',)

if mode=='GR':
    fitter_params = dict(files="0337+17-scott-2013-08-29",
                         parfile="0337_tempo2_pm.par",
                         tzrmjd_middle='auto',
                         fit_pos=True, fit_pm=False, fit_px=True,
                         t2_astrometry=True,
                         kopeikin=True,
                         ppn_mode='GR')
elif mode=='heavysimple':
    fitter_params = dict(files="0337+17-scott-2013-08-29",tzrmjd_middle='auto',
                         parfile="0337_tempo2_pm.par",
                         fit_pos=True,
                         fit_pm=False,
                         fit_px=True,
                         t2_astrometry=True,
                         kopeikin=True,
                         priors=('dbeta','dgamma'),
                         ppn_mode='heavysimple')
elif mode=='paper1':
    fitter_params = dict(files="0337+17-scott-2013-06-06",
                         parfile="0337_tempo2_nobinary.par",
                         tzrmjd_middle='auto',
                         fit_pos=False,
                         fit_pm=False,
                         fit_px=False,
                         t2_astrometry=True,
                         kopeikin=False,
                         priors=(),
                         ppn_mode=None)
elif mode=='paper2':
    fitter_params = dict(files="0337+17-scott-2013-06-06",
                         parfile="0337_tempo2_nobinary.par",
                         only_tels=only_tels,
                         tzrmjd_middle='auto',
                         fit_pos=False,
                         fit_pm=False,
                         fit_px=False,
                         t2_astrometry=True,
                         kopeikin=False,
                         priors=(),
                         ppn_mode=None)
elif mode=='vlbi-2014-02':
    fitter_params = dict(files="0337+17-anne-2014-02-04c",
                         parfile="0337_tempo2_px_optical.par",
                         tzrmjd_middle='auto',
                         fit_pos=True,
                         fit_pm=False,
                         fit_px=False,
                         t2_astrometry=True,
                         kopeikin=False,
                         priors=(),
                         ppn_mode='GR')
elif mode=='sep-2014-01':
    fitter_params = dict(files="0337+17-anne-2014-02-04c",
                         parfile="0337_tempo2_px_optical.par",
                         tzrmjd_middle='auto',
                         fit_pos=True,
                         fit_pm=False,
                         fit_px=False,
                         t2_astrometry=True,
                         kopeikin=False,
                         priors=('dbeta','dgamma'),
                         ppn_mode='heavysimple')
elif mode.startswith("file"):
    fn = mode.split(":")[1]
    fitter_params = pickle.load(open(fn,"rb"))
else:
    raise ValueError("Unknown mode")

F = threebody.Fitter(**fitter_params)
logger.debug("Fitter created")
j = 0
def lnprob(offset):
    global j
    logger.debug("call %d" % j)
    j = j+1
    params = F.best_parameters.copy()
    if len(offset)!=len(F.parameters):
        raise ValueError("Parameter mismatch between walker and Fitter")
    for p,o in zip(F.parameters, offset):
        params[p] += o
    logger.debug("started lnprob computation")
    r = F.lnprob(params)
    logger.debug("finished lnprob computation with %s" % r)
    extra_info = {}
    extra_info['linear_part'] = F.compute_linear_parts(params)
    for op in ['initial_values', 'time', 'n_evaluations', 'parameter_dict']:
        extra_info[op] = F.last_orbit[op]
    return r, extra_info
def lnprior(offset):
    params = F.best_parameters.copy()
    for p,o in zip(F.parameters, offset):
        params[p] += o
    return F.lnprior(params)
def lnprob_internal(offset):
    ll, blob = lnprob(offset)
    return ll+lnprior(offset), blob

logger.debug("creating pool")
pool = emcee.utils.MPIPool()
if not pool.is_master():
    logger.info("waiting for commands")
    pool.wait()
    sys.exit(0)

logger.info("ready to issue commands")

try:
    if trust_nfs:
        local_dbdir = dbdir
    else:
        local_dbdir = tempfile.mkdtemp()

    p0 = np.load("start-walkers.npy")
    logger.info("loaded walkers with dimension %s" % (p0.shape,))
    if p0.shape[0] == 1:
        logger.info("only one temperature so removing temperature axis")
        p0 = p0[0]
    if p0.shape[-1]!=len(F.parameters):
        raise ValueError("Parameter mismatch between "
            "walker (%dd) and Fitter (%dd)" % (p0.shape[-1],len(F.parameters)))

    if len(p0.shape)==2:
        logger.info("using EnsembleSampler (warning: untested)")
        sampler = emcee.EnsembleSampler(
            p0.shape[0],p0.shape[1],
            lnprob_internal,
            pool=pool) # FIXME: untested
    else:
        logger.info("using PTSampler (warning: can't handle blobs)")
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
            logger.debug("starting saving loop")
            while not done:
                save()
                time.sleep(30)
                logger.debug("saved at %s" % time.asctime())
        the_thread = threading.Thread(target=save_loop)
        the_thread.start()
    logger.debug("starting sampling loop")
    logger.info("fitter parameters %s" % fitter_params)
    with open(local_dbdir+"/fitter_params.pickle","wb") as f:
        pickle.dump(fitter_params,f)
    logger.info("parameters %s" % F.parameters)
    np.save(local_dbdir+"/parameters.npy", F.parameters)
    logger.info("best parameters %s" % F.best_parameters)
    np.save(local_dbdir+"/best_parameters.npy",
            np.array([F.best_parameters[p] for p in F.parameters]))
    logger.debug("starting sampling loop")
    i = 0
    for result in sampler.sample(p0, iterations=n_steps,
                                            storechain=False):
        pos = result[0]
        if len(p0.shape)==2:
            prob = result[1]
            blobs = np.array(result[3])
        else:
            prob = result[2]
            blobs = np.array(result[5])
        logger.debug("writing sample %d" % i)
        np.save(local_dbdir+"/%06d-pos.npy" % i, pos)
        np.save(local_dbdir+"/%06d-prob.npy" % i, prob)
        np.save(local_dbdir+"/%06d-blobs.npy" % i, blobs)
        i += 1
finally:
    if not trust_nfs:
        done = True
    pool.close()

