import tempfile
import os
import subprocess
import shutil

import numpy as np
import numpy.ma as ma


def approx_fit(stdprof, prof, oversample=16):
    if len(stdprof)!=len(prof):
        raise ValueError("Profile has length %d and standard has length %d but they should be equal" % (len(prof), len(stdprof)))

    std_scale = np.sqrt(np.sum((stdprof-np.mean(stdprof))**2))

    fs = np.fft.rfft(stdprof)
    f = np.fft.rfft(prof)

    ms = fs[0].real/len(stdprof)
    fs[0] = 0
    m = f[0].real/len(prof)
    f[0] = 0

    corr = np.fft.irfft(np.conj(fs)*f, len(stdprof)*oversample)
    ix = np.argmax(corr)
    phase = ix/float(len(corr))
    a = corr[ix]*oversample/std_scale**2
    b = m - a*ms

    fit_prof = np.fft.irfft(f*np.exp(2.j*np.pi*phase*np.arange(len(f))))+m
    fit_prof = (fit_prof-b)/a
    c = np.sqrt(np.mean((fit_prof-stdprof)**2))

    return phase, a, b, c, fit_prof

def downsample_by(p,f):
    return np.mean(p.reshape((-1,f)),axis=-1)
def align_profiles(files, std_np, ds_by=2):
    times = []
    profiles = []
    noises = []
    raw_profiles = []
    for F in files:
        pf = F.get_data()[:,0,0,:]
        for i in range(len(F)):
            times.append(F.get_Integration(i).get_start_time().in_days())
            pfi = pf[i]
            phase, a, b, c, fit_prof = approx_fit(std_np, pfi, oversample=64)
            #noises.append(np.amax(np.abs(fit_prof-std_np)))
            noises.append(np.sqrt(np.mean((fit_prof-std_np)**2)))
            profiles.append(downsample_by(fit_prof,ds_by))
            raw_profiles.append(downsample_by((pfi-b)/a,ds_by))
    times = np.array(times)
    profiles = np.array(profiles)
    raw_profiles = np.array(raw_profiles)
    noises = np.array(noises)
    ix = np.argsort(times)
    times = times[ix]
    profiles = profiles[ix]
    raw_profiles = raw_profiles[ix]
    noises = noises[ix]
    return times, profiles, raw_profiles, noises

def bin_profiles(bin_by, profiles, n_bins=128, bin_range=None, weights=None):
    bin_by = np.asarray(bin_by)
    profiles = np.asarray(profiles)
    n, b = profiles.shape
    if bin_by.shape != (n,):
        raise ValueError("bin_by shape %s does not match profiles shape %s"
                         % (bin_by.shape, profiles.shape))

    r = np.zeros((n_bins,b))
    w = np.zeros(n_bins)

    if bin_range is None:
        bin_range = np.amin(bin_by), np.amax(bin_by)
    for i in range(n):
        ix = int(n_bins*(bin_by[i]-bin_range[0])/(bin_range[1]-bin_range[0]))
        if 0<=ix<n_bins:
            if weights is None:
                wi = 1
            else:
                wi = weights[i]
            r[ix] += wi*profiles[i]
            w[ix] += wi
    m = ma.masked_invalid(r/w[:,None])
    return m

def mostly_range(a, frac_out=0.05):
    ac = ma.masked_array(a).compressed().copy()
    ac.sort()
    return ac[(frac_out/2)*len(ac)], ac[-(frac_out/2)*len(ac)]

def barycenter_times(times, par_file="0337_bogus.par"):
    bbats = []
    d = tempfile.mkdtemp()
    try:
        i = 0
        block = 9000
        while i<len(times):
            tf = os.path.join(d,"file.tim")
            with open(tf,"wt") as f:
                f.write("FORMAT 1\n")
                for t in times[i:i+block]:
                    f.write("fake 999999.999 %s 3.00 @\n" % t)
            o = subprocess.check_output(["tempo2",
                                         "-output", "general2",
                                         "-s", "OUTPUT {bbat}\n",
                                         "-f",par_file,
                                         tf])
            for l in o.split("\n"):
                if not l.startswith("OUTPUT"):
                    continue
                _, bbat = l.split()
                bbat = float(bbat)
                bbats.append(bbat)
            i += block
    finally:
        shutil.rmtree(d)
    return np.array(bbats)
