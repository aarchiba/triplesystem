from __future__ import division
import os, shutil, sys
import logging
import pickle
import numpy as np
import scipy.stats

import pymultinest

import threebody

jobdir, = sys.argv[1:]
jobbase = os.path.join(jobdir,"chain")

fitter_params = pickle.load(
    open(os.path.join(jobdir,"multinest_params.pickle"),"rb"))
F = threebody.Fitter(**fitter_params)

n_params = len(F.parameters)

plotter = pymultinest.watch.ProgressPlotter(n_params, interval_ms=1e4,
                                            outputfiles_basename=jobbase)
plotter.start()

printer = pymultinest.watch.ProgressPrinter(n_params, interval_ms=1e4,
                                            outputfiles_basename=jobbase)
printer.start()
