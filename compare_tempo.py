import subprocess
import numpy as np
import matplotlib.pyplot as plt
import residual

par_t2_tdb = """
PSR              J0337+17
RAJ      03:37:43.826099
DECJ      17:15:14.826651
PX                    0.77
PMRA                  3.56
PMDEC                -3.90
F0    365.9533436144258189  0  0.0000000000656904
F1      7.833539631670D-15  0  1.490959143049D-17
PEPOCH        56500.000000
DM               21.313000
SOLARN0              10.00
EPHEM             DE421
TZRMJD  56100.13622674904492
TZRFRQ            1379.999
TZRSITE                  j
UNITS TDB
"""

def tempo_resids(par):
    temppar = "temp.par"
    with open(temppar, "wt") as f:
        f.write(par)
    subprocess.check_call(["tempo", "-f", temppar, compare_tim])
    r = residuals.read_residuals()
    return r.bary_TOA, r

def tempo2_resids(par):
    temppar = "temp.par"
    with open(temppar, "wt") as f:
        f.write(par)
    o = subprocess.check_output(
        ["tempo2",
         "-output", "general2",
         "-s", "OUTPUT {bat}\n",
         "-f", temppar,
         compare_tim])
    r = []
    for l in o.split("\n"):
        if not l.startswith("OUTPUT"):
            continue
        r.append(float(l.split()[1]))
    return np.array(r)

r1, r = tempo_resids(par_t2_tdb)

r3 = tempo2_resids(par_t2_tdb)

plt.plot(r1, (r3-r1)*86400*1e6,
         ".",markersize=1)
plt.ylabel(r"$\mu$s")
plt.xlabel("MJD")
plt.title("tempo2 barycentered TOAs minus tempo")
plt.savefig("barydiff-simple.pdf")
plt.show()
