#!/usr/bin/env python

import re
import sys
import argparse
import subprocess
import shutil

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("MJD", help="MJD of the observation", type=float)
parser.add_argument("--length", help="Length of the orbital segment to fit (days)",
        default=10., type=float)
parser.add_argument("--toafile", help="File listing simulated TOAs",
        default="fake.tim")
parser.add_argument("--pulsesfile", help="File listing pulse numbers (must be exactly one per fake TOA)",
        default="fake.pulses")
parser.add_argument("--tempparfile", help="par file to create and ask TEMPO to fit",
        default="temposculating.par")
parser.add_argument("--temptim", help="tim file containing segment",
        default="temp.tim")
parser.add_argument("--temppulses", help="tim file containing segment",
        default="temp.pulses")
parser.add_argument("--oscfilename",
        help="File listing osculating orbits", default="osculating.txt")
parser.add_argument("--fixedbtx",
        help="BTX model includes Doppler correction", action="store_true")
parser.add_argument("--zerofirst",
        help="First pulse is number zero", action="store_true")

args = parser.parse_args()


n = 0
np0 = 0
with open(args.temptim, "wt") as temptim:
    with open(args.temppulses, "wt") as temppulses:
        for toaline, pulseline in zip(open(args.toafile,"rt").readlines(),
                                      open(args.pulsesfile,"rt").readlines()):
            if not toaline:
                continue
            toa = float(toaline[24:44])
            if args.MJD-args.length/2<=toa<=args.MJD+args.length/2:
                temptim.write(toaline)
                if n==0:
                    np0 = int(pulseline)
                if args.zerofirst:
                    temppulses.write(str(int(pulseline)-np0)+"\n")
                else:
                    temppulses.write(pulseline)
                n += 1
if n==0:
    raise ValueError("Input MJD past end or before beginning of simulated TOAs")

osculating_parameters = np.loadtxt(args.oscfilename)
i = np.searchsorted(osculating_parameters[:,0],args.MJD)
if i==len(osculating_parameters):
    i -= 1
elif i<len(osculating_parameters)-1 and osculating_parameters[i+1,0]-args.MJD<args.MJD-osculating_parameters[i,0]:
    i += 1
col_names = open(args.oscfilename).readline().split()[1:]
d = dict(zip(col_names,osculating_parameters[i]))

if args.fixedbtx:
    d['model'] = "BTX"
else:
    d['model'] = "BTX"
    d['t0_i'] = d['t0_i']+d['z_i']/86400
    d['pb_i'] = d['pb_i']*(1+d['vz_i']/86400.)

template_noastro = """PSR              J0337+17
RAJ      03:37:43.82589000
DECJ      17:15:14.8281000
F0                  {f0!r} 1
F1                  {f1!r} 1
PEPOCH        56100.000000
DM               21.313000
SOLARN0              10.00
EPHEM             DE405
CLK               UTC(NIST)
TZRMJD  56100.13622674904489
TZRFRQ            1379.999
TZRSITE                  @
NITS                     1
BINARY             {model}
PLAN  1
A1             {asini_i!r} 1
E                  {e_i!r} 1
T0                {t0_i!r} 1
OM                {om_i!r} 1
PB                {pb_i!r} 1
A1_2           {asini_o!r}
E_2                {e_o!r}
T0_2              {t0_o!r} 1
PB_2              {pb_o!r}
OM_2              {om_o!r}
"""
template = """PSR              J0337+17
RAJ      03:37:43.826099
DECJ      17:15:14.826651
PX                    1.40
PMRA                  3.56
PMDEC                -3.90
F0                  {f0!r} 1
F1                  {f1!r} 1
POSEPOCH 56500
PEPOCH        56100.000000
DM               21.313000
SOLARN0              10.00
EPHEM             DE405
CLK               UTC(NIST)
TZRMJD  56100.13622674904489
TZRFRQ            1379.999
TZRSITE                  @
NITS                     1
BINARY             {model}
PLAN  1
A1             {asini_i!r} 1
E                  {e_i!r} 1
T0                {t0_i!r} 1
OM                {om_i!r} 1
PB                {pb_i!r} 1
A1_2           {asini_o!r}
E_2                {e_o!r}
T0_2              {t0_o!r} 1
PB_2              {pb_o!r}
OM_2              {om_o!r}
"""
with open(args.tempparfile, "wt") as f:
    f.write(template.format(**d))

n = 0
while True:
    o = subprocess.check_output(["tempo",
            "-ni",args.temppulses,
            "-f",args.tempparfile,
            args.temptim],
            stderr=subprocess.STDOUT)

    l = o.split("\n")[-2]
    print l
    m = re.search(r"[Pp]re-fit\s*((\d+.\d+)|NaN)+\s+us", l)
    error = float(m.group(1))
    print error
    break
    if error<10:
        break
    if n>10:
        raise ValueError("TEMPO does not appear to be converging")
    n += 1
    shutil.copy("J0337+17.par", args.tempparfile)

print "Results should be in J0337+17.par"
