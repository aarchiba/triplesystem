#!/usr/bin/env python

from __future__ import division, print_function
import argparse
import pickle
import numpy as np

import threebody
import numdifftools

# default 1e-3
base_steps = dict(asini_i=1e-4, 
                  pb_i=1e-7, 
                  eps1_i=1e-4, 
                  eps2_i=1e-4,
                  tasc_i=1e-4,
                  pb_o=1e-4,
                  delta=1e-7,
                  eps2_o=1e-5,
                  eps1_o=1e-5,
                  acosi_o=0.1,
                  asini_o=1e-5,
                  dbeta=0.01,
                  dgamma=0.01,
                  )

def derivative(params, p, output, best_parameters=None, partial=False):
    F = threebody.Fitter(**params)
    if p not in F.parameters:
        raise ValueError("Parameter %s not among options: %s" % (p, F.parameters))
    if best_parameters is not None:
        F.best_parameters = pickle.load(open(best_parameters,"rb"))
    print(F.goodness_of_fit(F.best_parameters))
    if partial:
        vals, names = F.compute_linear_parts()
        for (n,v) in zip(names,vals):
            F.best_parameters[n] = v
    bs = base_steps.get(p, 1e-3)
    sg = numdifftools.MaxStepGenerator(bs,
                                       use_exact_steps=True)
    def fres(v):
        bp  = F.best_parameters.copy()
        print(p, bp[p], v, end="\t")
        bp[p] += v
        try:
            r = F.residuals(bp, linear_fit=False)
            print(F.goodness_of_fit(bp))
            return r
        except ValueError:
            print(np.inf)
            return np.inf*np.ones_like(F.mjds)
    nl_der = numdifftools.Derivative(fres, step=sg)(0)
    np.save(output, nl_der)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Compute derivatives of a triplesystem model")
    parser.add_argument("params", 
        help="pickle containing the fitter options")
    parser.add_argument("param_name", 
        help="the parameter to compute the derivative with respect to")
    parser.add_argument("output", 
        help="where to store the output derivative")
    parser.add_argument("--best_parameters",
        help="pickle containing starting values for fitter; "
             "defaults to what's in the database")
    parser.add_argument("--partial",
        help="compute true partial derivatives "
             "(that is, hold the linear parts fixed)",
                        action="store_true")
    args = parser.parse_args()

    params = pickle.load(open(args.params,"rb"))
    derivative(params, args.param_name, args.output, 
               best_parameters=args.best_parameters,
               partial=args.partial)


