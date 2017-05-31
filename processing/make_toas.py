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

import numpy as np
import scipy.optimize
from numpy.fft import rfft, irfft, fft, ifft

import psrchive
import residuals

import joblib
import pipe

processing_specs = pickle.load(open("processing_specs.pickle","rb"))
toa_specs = pickle.load(open("toa_specs.pickle","rb"))

class NoSpecError(pipe.ProcessingError):
    pass

def generate(observation, processing_name, toa_name):
    import matplotlib.pyplot as plt
    if not os.path.exists(join(observation, processing_name)):
        meta = pickle.load(open(join(observation,"meta.pickle"),"rb"))
        spec = processing_specs[processing_name]
        kwargs = spec["generic"].copy()
        k = meta["tel"], meta["band"]
        if k not in spec:
            raise NoSpecError(
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
            raise NoSpecError(
                "No TOA spec for %s in %s" % (k, toa_name))
        kwargs.update(spec[k])
        pipe.make_toas(observation, processing_name, toa_name, **kwargs)
	plt.close('all')

def generate_specific(observation, processing_name, toa_name, trybad=False, stop=False):
    print observation, processing_name, toa_name
    kb = join(observation,"knownbad")
    if os.path.exists(kb):
        print "known bad, explanation:", open(kb,"rt").read()
        if not trybad:
            return
    try:
        generate(observation, processing_name, toa_name)
    except NoSpecError as e:
        print e
    except pipe.ProcessingError as e:
        if stop:
            raise
        print e

if __name__=='__main__':
    matplotlib.use('PDF')
    matplotlib.rcParams['savefig.dpi'] = 144
    matplotlib.rcParams["image.composite_image"]=False

    import matplotlib.pyplot as plt
    plt.viridis()


    parser = argparse.ArgumentParser()
    parser.add_argument("processing_name", help="Name of a processing setup to use")
    parser.add_argument("toa_name", help="Name of a TOA-generating setup to use")
    parser.add_argument("--njobs",
                        help="Number of parallel jobs to run (using joblib)",
                        type=int,
                        default=1)
    parser.add_argument("--trybad",
                        help="Try to process observations marked 'knownbad'",
                        action="store_true")
    parser.add_argument("--stop",
                        help="Stop if an error occurs",
                        action="store_true")
    args = parser.parse_args()


    processing_name = args.processing_name
    toa_name = args.toa_name

    if processing_name not in processing_specs:
        raise ValueError("Unknown processing_name %s; possibilities are %s"
                         % (processing_name, sorted(processing_specs.keys())))
    if toa_name not in toa_specs:
        raise ValueError("Unknown toa_name %s; possibilities are %s"
                         % (toa_name, sorted(toa_specs.keys())))

    print processing_name, toa_name

    observations = sorted(glob("data/obs/*_*_*"))

    if args.njobs==1:
        for o in observations:
            generate_specific(o, processing_name, toa_name, trybad=args.trybad, stop=args.stop)
    else:
        Parallel(n_jobs=args.njobs)(delayed(generate_specific)(o, processing_name, toa_name, 
                                                               trybad=args.trybad, stop=args.stop)
                                           for o in observations)
    print "All observations tried."
