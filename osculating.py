#!/usr/bin/env python

import sys
import argparse

import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("MJD", 
    help="Modified Julian Day to center the osculating orbit on (actually, use the nearest orbit in the data file)", type=float)
parser.add_argument("--filename", 
    help="File listing osculating orbits", default="osculating.txt")
parser.add_argument("--fixedbtx", 
    help="BTX model includes Doppler correction", action="store_true")

args = parser.parse_args()

MJD = float(args.MJD)

osculating_parameters = np.loadtxt(args.filename)
i = np.searchsorted(osculating_parameters[:,0],MJD)
if osculating_parameters[i+1,0]-MJD>MJD-osculating_parameters[i,0]:
    i += 1
col_names = open(args.filename).readline().split()[1:]
d = dict(zip(col_names,osculating_parameters[i]))

if args.fixedbtx:
    d['model'] = "BTX"
else:
    d['model'] = "BT1P"
    d['t0_i'] = d['t0_i']+d['z_i']/86400
    d['pb_i'] = d['pb_i']*(1+d['vz_i']/86400.)
    # FIXME: can OMDOT help? act like T0dot, modified by vz_i

template = """PSR              J0337+17    
RAJ      03:37:43.82100000
DECJ      17:15:14.8200000
F0                  {f0!r}
F1                  {f1!r}
PEPOCH        56100.000000
DM               21.313000
SOLARN0              10.00
EPHEM             DE405
CLK               UTC(NIST)   
TZRMJD  56100.13622674904489
TZRFRQ            1379.999
TZRSITE                  j
NITS                     1
BINARY             {model}     
PLAN  1
A1             {asini_i!r}
E                  {e_i!r}
T0                {t0_i!r}
OM                {om_i!r}
PB                {pb_i!r}
A1_2           {asini_o!r}
E_2                {e_o!r}
T0_2              {t0_o!r}
PB_2              {pb_o!r}
OM_2              {om_o!r}
"""

print template.format(**d)
