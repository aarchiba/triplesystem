from __future__ import division
import os, shutil
import numpy as np

import pymultinest

print "Started python"

S = np.load("multinest-test-setup.npz")
xs = S['xs']
ys = S['ys']
ndim = S['ndim']
noise = S['noise']
prior_scale = S['prior_scale']

if ndim%2:
    raise ValueError
    
true_parameters = np.repeat((np.arange(ndim//2)+1.)**(-1),2)
true_parameters[::2] = 0

def model(parameters):
    # aargh parameters is not a numpy array
    ick_parameters = parameters
    parameters = np.zeros(ndim,dtype=float)
    parameters[:] = ick_parameters[:len(parameters)]
    ft = np.zeros(len(parameters)//2+1, dtype=np.complex)
    ft[1:] = parameters[::2]+1.j*parameters[1::2]
    return np.fft.irfft(ft,len(xs))*len(xs)

def prior(cube, ndim, nparams):
    for i in range(ndim):
        cube[i]= (2*cube[i]-1)*prior_scale
    
def loglike(cube, ndim, nparams):
    mys = model(cube)
    return -0.5*np.sum(((mys-ys)/noise)**2)

outdir = 'multinest-testing/mpi/'
if not os.path.exists(outdir):
    os.makedirs(outdir)

print "Calling into multinest"

pymultinest.run(loglike, prior, len(true_parameters), 
                outputfiles_basename=outdir,
                init_MPI = False,
                resume = False, verbose = True)
