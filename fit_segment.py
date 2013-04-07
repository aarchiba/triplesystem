
import sys
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("MJD", help="MJD of the observation", type=float)
parser.add_argument("--length", help="Length of the orbital segment to fit (days)", 
        default=3., type=float)
parser.add_argument("--toafile", help="File listing simulated TOAs",
        default="fake.tim")
parser.add_argument("--pulsesfile", help="File listing pulse numbers (must be exactly one per fake TOA)",
        default="fake.pulses")
parser.add_argument("--parfile", help="par file to ask TEMPO to fit",
        default="segment.par")
parser.add_argument("--temptim", help="tim file containing segment",
        default="temp.tim")
parser.add_argument("--temppulses", help="tim file containing segment",
        default="temp.pulses")

args = parser.parse_args()

n = 0
with open(args.temptim, "wt") as temptim:
    with open(args.temppulses, "wt") as temppulses:
        for toaline, pulseline in zip(open(args.toafile,"rt").readlines(),
                                      open(args.pulsesfile,"rt").readlines()):
            toa = float(toaline[24:44])
            if args.MJD-args.length/2<=toa<=args.MJD+args.length/2:
                temptim.write(toaline)
                temppulses.write(pulseline)
                n += 1
if n==0:
    raise ValueError("Input JD past end of simulated TOAs")

subprocess.check_call(["tempo",
    "-ni",args.temppulses,
    "-f",args.parfile,
    args.temptim])

