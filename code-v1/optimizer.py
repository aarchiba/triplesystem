from glob import glob
import os
import sys
import tempfile
import subprocess

import numpy as np
import scipy.linalg

def fit_quadratic(ps, chi2s):
    m = len(ps)
    n = len(ps[0])
    A = []
    for p in ps:
        const = [1]
        deriv = p
        quad = []
        for i in range(n):
            for j in range(i,n):
                quad.append(p[i]*p[j])
        A.append(np.concatenate([const,deriv,quad]))
    coeffs, res, rk, s = scipy.linalg.lstsq(A,chi2s)
    const = coeffs[0]
    deriv = coeffs[1:n+1]
    quad = np.zeros((n,n))
    k = n+1
    for i in range(n):
        for j in range(i,n):
            quad[i,j] += 0.5*coeffs[k]
            quad[j,i] += 0.5*coeffs[k]
            k += 1
    return const, deriv, quad

def new_shape(ps, chi2s):
    m,n = np.shape(ps)
    const, deriv, quad = fit_quadratic(ps,chi2s)

    new_min = scipy.linalg.solve(quad,-deriv/2.)

    vals, vecs = scipy.linalg.eigh(quad)


    return new_min, vals, vecs

def gen_samples(m, new_min, vals, vecs):
    n = len(vals)

    vals = np.array(vals)
    minval = 1e-10*np.amax(vals)
    if vals[0]<minval:
        vals += (minval-vals[0])

    new_min = np.squeeze(new_min)

    new_ps = []
    for i in range(m):
        new_ps.append(
            np.squeeze(new_min+
                       np.squeeze(
                           np.dot(vecs,
                                  vals**(-0.5)*np.random.randn(n)/np.sqrt(n)))))
    return new_ps

def optimize(ps, func, pool=None, explore=1, adapt_covariance=True):
    m, n = np.shape(ps)
    ps = list(ps)
    assert len(ps)==m
    assert np.shape(ps[0])==(n,)
    if pool is None:
        M = map
    else:
        M = pool.map
    best_chi2 = np.inf
    best_pos = None
    no_improvement = 0
    original_vals, original_vecs = scipy.linalg.eigh(np.cov(ps,rowvar=0))
    while True:
        print "Computing values for an array of shape", np.shape(ps)
        chi2s = M(func,ps)
        print "Looking for improvement"
        no_improvement += 1
        for p,c in zip(ps,chi2s):
            if c<best_chi2:
                best_chi2 = c
                best_pos = np.squeeze(p)
                no_improvement = 0
        yield best_pos, best_chi2, ps, chi2s
        print "Choosing new search points"
        new_min, vals, vecs = new_shape(ps, chi2s)
        #print "vals:", vals
        #print "vecs:", vecs
        if not adapt_covariance:
            vals = original_vals
            vecs = original_vecs
        if no_improvement>explore:
            ps = gen_samples(m, best_pos, vals, vecs)
            no_improvement = 0
        else:
            ps = gen_samples(m, new_min, vals, vecs)
            ps[0] = new_min
        ps = [np.squeeze(p) for p in ps]
        print ps
