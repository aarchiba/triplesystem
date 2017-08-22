#!/usr/bin/env python

from __future__ import division, print_function
import argparse
import pickle

import threebody
import minuit_wrap

def optimize(params, best_filename, best_parameters=None, method="BOBYQA", output=None):
    F = threebody.Fitter(**params)
    if (best_parameters is not None and best_parameters!="default"):
        F.best_parameters = pickle.load(open(best_parameters,"rb"))
    M = minuit_wrap.Fitter(F.make_mfun(), output=output)
    for p in F.parameters:
        M.values[p] = F.best_parameters[p]
        M.errors[p] = F.best_errors[p]
    M.set_normalization()
    M.tol = 100
    M.strategy=1
    M.printMode = 3
    M.best_filename = best_filename

    if method=="MINUIT":
        M.migrad()
    elif method=="Powell":
        M.scipy_minimize("Powell", options=dict(ftol=1e-8))
    elif method=="basinhopping":
        M.scipy_minimize("basinhopping", 
            minimizer_kwargs=dict(
                method="Powell", 
                options=dict(ftol=1e-6)))
    elif method=="BOBYQA":
        M.nlopt_minimize()
    elif method=="simplex":
        import nlopt
        M.nlopt_minimize(optimizer=nlopt.LN_SBPLX)
    elif method=="gp":
        M.gp_minimize(n_random_starts=len(F.parameters),
                      n_calls=1000, verbose=True)
    else:
        raise ValueError("Unknown method")



if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Optimize a triplesystem model")
    parser.add_argument("params", 
        help="pickle containing the fitter options")
    parser.add_argument("best_filename", 
        help="where to store the best values found by the optimizer")
    parser.add_argument("--best_parameters",
        help="pickle containing starting values for fitter; "
             "defaults to what's in the database")
    parser.add_argument("--method", 
                        choices=["BOBYQA", "MINUIT", "Powell", "basinhopping", "gp", "simplex"], 
                        default="BOBYQA",
                        help="optimizer method")
    args = parser.parse_args()

    params = pickle.load(open(args.params,"rb"))
    optimize(params, args.best_filename, 
             best_parameters=args.best_parameters,
             method=args.method)
