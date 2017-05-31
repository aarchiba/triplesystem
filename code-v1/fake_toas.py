#!/usr/bin/env python
import os
import cPickle as pickle

import numpy as np
import scipy.linalg

import kepler
import quad_integrate
import threebody

fp = "fake_toas_fitter.pickle"
if os.path.exists(fp):
    d = pickle.load(open(fp,"rb"))
    F = threebody.Fitter(**d)
else:
    F = threebody.Fitter()

rms = 3e-6
days = 1000
n_per_day = 100
times = np.linspace(1,days+1,n_per_day*days+1)

o = F.compute_orbit(F.best_parameters)
F_linear = F.compute_linear_parts(F.best_parameters)

def freq(t):
    t0 = F_linear['tzrmjd']+(F_linear['tzrmjd_base']-F.base_mjd)
    t_s = (t-t0)*86400
    return (F_linear['f0']
            +F_linear['f1']*t_s)
def phase(t):
    t0 = F_linear['tzrmjd']+(F_linear['tzrmjd_base']-F.base_mjd)
    t_s = (t-t0)*86400
    return (F_linear['f0']*t_s
            +F_linear['f1']*t_s**2/2.)

with open("fake.tim","wt") as tim:
    with open("fake-t2.tim","wt") as tim2:
        tim2.write("FORMAT 1\n")
        with open("fake.pulses","wt") as pulses:
            for i,t_bb in enumerate(o["t_bb"]):
                t_psr = o["t_psr"][i]
                p = phase(t_psr)
                pulse = np.round(p)
                dt = -(p-pulse)*freq(t_psr)/86400.
                t_bb = t_bb + dt
                # now t is a barycentered arrival time
                toaline = ("@             999999.999 %05d.%s%9.2f\n" %
                    (F.base_mjd+int(np.floor(t_bb)),
                     ("%.13f" % (t_bb-np.floor(t_bb)))[2:15],
                     rms*1e6))
                tim.write(toaline)
                pulses.write("%d\n" % pulse)

                toaline2 = ("fake 999999.999 %05d.%s %9.2f @ -npulse %d\n" %
                    (F.base_mjd+int(np.floor(t_bb)),
                     ("%.13f" % (t_bb-np.floor(t_bb)))[2:15],
                     rms*1e6,
                     pulse))
                tim2.write(toaline2)
