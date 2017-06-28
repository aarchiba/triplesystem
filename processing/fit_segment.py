#!/usr/bin/env python

import re
import sys
import argparse
import subprocess
import shutil

import numpy as np

parser = argparse.ArgumentParser(
    description="Fit a par file to a short segment of the triple system orbit",
    epilog="""WARNING:

The output ephemeris only works with tempo 1. Tempo2 does not support the
two-orbit BTX model needed, and so it silently ignores this model. So if you
are generating polyco's, use tempo 1. If you are using PSRCHIVE, it usually
defaults to using tempo2, which will result in silently misaligned files.
To force PSRCHIVE to use tempo1, place the following lines in your
$HOME/.psrchive.cfg:

Predictor::default = polyco
Predictor::policy = default

These force the use of tempo 1, so you may need to remove them if you want to
work with tempo2 ephemerides. This unsatisfying solution appears to be the
best option currently available.
""")
parser.add_argument("MJD", help="MJD of the observation", type=float)
parser.add_argument("--length",
        help="Length of the orbital segment to fit (days)",
        default=10., type=float)
parser.add_argument("--toafile", help="File listing simulated TOAs",
        default="fake.tim")
parser.add_argument("--pulsesfile",
        help="File listing pulse numbers (must be exactly one per fake TOA)",
        default="fake.pulses")
parser.add_argument("--tempparfile",
        help="par file to create and ask TEMPO to fit",
        default="temposculating.par")
parser.add_argument("--temptim", help="tim file containing segment",
        default="temp.tim")
parser.add_argument("--temppulses", help="tim file containing segment",
        default="temp.pulses")

args = parser.parse_args()


n = 0
tzrmjd = None
with open(args.temptim, "wt") as temptim:
    with open(args.temppulses, "wt") as temppulses:
        for toaline, pulseline in zip(open(args.toafile,"rt").readlines(),
                                      open(args.pulsesfile,"rt").readlines()):
            if not toaline:
                continue
            toa = float(toaline[24:44])
            if args.MJD-args.length/2<=toa<=args.MJD+args.length/2:
                temptim.write(toaline)
                if tzrmjd is None:
                    tzrmjd = toaline.split()[2]
                temppulses.write(pulseline)
                n += 1
if n==0:
    raise ValueError("Input MJD past end or before beginning of simulated TOAs")

print "input TZRMJD:", tzrmjd

fit_par = """PSR              J0337+17
RAJ      03:37:43.82589000
DECJ      17:15:14.8281000
POSEPOCH        56337.0000
F0    365.9533436437517366  1  0.0000000000025982
F1      3.459285698888D-16  1  5.908704611672D-19
PEPOCH        56100.000000
DM               21.315933
SOLARN0              10.00
CLK               UTC(NIST)
NTOA                 26296
TRES                 38.47
TZRMJD            {tzrmjd}
TZRSITE                  @
TZRFREQ         999999.999
BINARY                 BTX
PLAN  1
A1             1.217528496  1         0.000000010
E             0.0006802884  1        0.0000000154
T0         55917.574547351  1         0.000006180
OM         94.157681118988  1      0.001365536908
FB0     7.103160919367D-06  1  2.040868704844D-16
A1_2          74.668594264
E_2            0.035347540
T0_2       56317.235447282  1         0.000000705
PB_2      327.219804954111
OM_2       95.726944226135
"""
with open(args.tempparfile, "wt") as f:
    f.write(fit_par.format(tzrmjd=tzrmjd))

n = 0
while True:
    o = subprocess.check_output(["tempo",
            "-ni",args.temppulses,
            "-f",args.tempparfile,
            args.temptim],
            stderr=subprocess.STDOUT)

    l = o.split("\n")[-2]
    print l
    m = re.search(r"[Pp]re-fit\s+(\d+.\d+)+\s+us", l)
    if m is not None:
        error = float(m.group(1))
        print error
        if error<5:
            break
    if n>5:
        raise ValueError
    n += 1
    shutil.copy("J0337+17.par", args.tempparfile)

print "Results should be in J0337+17.par"
