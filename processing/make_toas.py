import os
from glob import glob
import subprocess
import shutil
import traceback
import random
import pickle
from os.path import join
import argparse

import joblib
from joblib import Parallel, delayed
from concurrent.futures import ProcessPoolExecutor

from backports import tempfile

import matplotlib
matplotlib.use('PDF')
matplotlib.rcParams['savefig.dpi'] = 144
matplotlib.rcParams["image.composite_image"]=False

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from numpy.fft import rfft, irfft, fft, ifft

import psrchive
import residuals

import joblib
import pipe

plt.viridis()


parser = argparse.ArgumentParser()
parser.add_argument("processing_name", help="Name of a processing setup to use")
parser.add_argument("toa_name", help="Name of a TOA-generating setup to use")
parser.add_argument("--njobs",
                    help="Number of parallel jobs to run (using joblib)",
                    type=int,
                    default=1)
args = parser.parse_args()


processing_name = args.processing_name
toa_name = args.toa_name

processing_specs = pickle.load(open("processing_specs.pickle","rb"))
toa_specs = pickle.load(open("toa_specs.pickle","rb"))

processing_specs[processing_name]
toa_specs[toa_name]

print processing_name, toa_name


def generate(observation, processing_name, toa_name):
    if not os.path.exists(join(observation, processing_name)):
        meta = pickle.load(open(join(observation,"meta.pickle"),"rb"))
        spec = processing_specs[processing_name]
        kwargs = spec["generic"].copy()
        k = meta["tel"], meta["band"]
        if k not in spec:
            raise pipe.ProcessingError(
                "No processing spec for %s in %s" % (k, processing_name))
        kwargs.update(spec[k])
        pipe.process_observation(observation, processing_name, **kwargs)
    if not os.path.exists(join(observation, processing_name, toa_name)):
        meta = pickle.load(open(join(
            observation,processing_name,"process.pickle"),"rb"))
        spec = toa_specs[toa_name]
        kwargs = spec["generic"].copy()
        k = meta["tel"], meta["band"]
        if k not in spec:
            raise pipe.ProcessingError(
                "No TOA spec for %s in %s" % (k, toa_name))
        kwargs.update(spec[k])
        pipe.make_toas(observation, processing_name, toa_name, **kwargs)

def generate_specific(observation):
    print observation, processing_name, toa_name
    try:
        generate(observation, processing_name, toa_name)
    except pipe.ProcessingError as e:
        print e

observations = sorted(glob("data/obs/*_*_*"))

if args.njobs==1:
    for o in observations:
        generate_specific(o)
else:
    Parallel(n_jobs=args.njobs)(delayed(generate_specific)(o)
                                       for o in observations)
print "All observations tried."
