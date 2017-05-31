import sys
import os
import cPickle as pickle

import numpy as np
import minuit

import threebody
import minuit_wrap

def make_fitter(d):
    return threebody.Fitter(**d)


def wrap_fitter(F):
    M = minuit_wrap.Fitter(F.make_mfun())
    for p in F.parameters:
        M.values[p] = F.best_parameters[p]
        M.errors[p] = F.best_errors[p]
    M.set_normalization()
    M.maxcalls = 10000
    M.tol = 100
    M.eps = 1e-6
    M.strategy = 2
    #M.printMode = 3

    return M

def main(output_dir,arrayid,fitter_params,bootstrap_params):

    F = make_fitter(fitter_params)

    param_file = os.path.join(output_dir,"parameters.pickle")
    fitter_file = os.path.join(output_dir,"fitter.pickle")
    bootstrap_file = os.path.join(output_dir,"bootstrap.pickle")

    if os.path.exists(fitter_file):
        p = pickle.load(open(fitter_file,"rb"))
        if p!=fiter_params:
            raise ValueError("Running bootstrap in directory with parameters %s but script parameters are %s" % (p,fitter_params))
    else:
        with open(fitter_file, "wb") as f:
            pickle.dump(fitter_params, f)
    with open(param_file, "wb") as f:
        pickle.dump(F.parameters, f)
    with open(bootstrap_file, "wb") as f:
        pickle.dump(bootstrap_params, f)

    ix = -1
    while True:
        ix += 1
        out_base = os.path.join(output_dir,"boot-%03d-%07d-" % (arrayid,ix))
        mjd_file = out_base+"mjds.npy"
        state_file = out_base+"state.txt"
        result_file = out_base+"params.npy"
        fval_file = out_base+"fval.npy"
        ncalls_file = out_base+"ncalls.npy"
        if os.path.exists(mjd_file):
            continue
        F = make_fitter(fitter_params)
        F.bootstrap()
        if os.path.exists(mjd_file):
            continue
        np.save(mjd_file, F.mjds)

        M = wrap_fitter(F)
        try:
            if bootstrap_params['minimizer']=='minuit':
                M.strategy = bootstrap_params['strategy']
                M.migrad()
            elif bootstrap_params['minimizer']=='simplex':
                M.simplex()
            if M.ncalls == M.maxcalls:
                with open(state_file, "wt") as f:
                    f.write("maxcalls\n")
            else:
                with open(state_file, "wt") as f:
                    f.write("success\n")
        except minuit.MinuitError:
            with open(state_file, "wt") as f:
                f.write("exception\n")
        params = np.array([M.best_values[p] for p in F.parameters])
        np.save(result_file, params)
        np.save(fval_file, M.best_values_fval)
        np.save(ncalls_file, M.ncalls)

if __name__=='__main__':
    jobid = os.environ.get('PBS_JOBID','local')
    dbdir = os.path.join('/home/aarchiba/projects/threebody/bootstrap-runs',jobid)
    os.mkdir(dbdir)
    arrayid = int(os.environ.get('PBS_ARRAYID'))
    fitter_params = dict(files="0337+17-scott-2013-08-29",
                         tzrmjd_middle='auto',
                         parfile="0337_tempo2_pm.par",
                         fit_pos=True,
                         fit_pm=False,
                         fit_px=True,
                         t2_astrometry=True,
                         kopeikin=True,
                         priors=('dbeta','dgamma'),
                         ppn_mode='heavysimple')
    bootstrap_params = dict(minimizer="minuit",
                            strategy=1)
    #bootstrap_params = dict(minimizer="simplex")
    main(dbdir,arrayid,fitter_params,bootstrap_params)
