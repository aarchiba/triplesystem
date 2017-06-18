#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A set of tools for computing phase shifts between templates and data

A standard operation in pulsar astronomy is to take an observation of
a pulsar and compute the phase shift between a standard template and
the observed data. This allows pulsar timing to be carried out based on
the information of when the pulse from the pulsar actually arrived.

Standard algorithms are based purely upon the intensity part of the
signal. Telescope calibration is rarely reliable enough to try to
match the polarization components, though they may contain additional
information to constrain the relative phase. This code distrusts the
polarization calibration completely, fitting for an arbitrary Mueller
matrix to transform the template into something that approximates the
data. Since the polarization calibration fit is a linear least-squares
fit, it doesn't even require an initial guess, and can accommodate
(for example) sign flips or loss of a polarization channel.

This file is intended to be used in two ways: as a command-line program
to generate TOAs as part of a pipeline, and as a python module for
computing the same quantities along with all sorts of diagnostic
information.
"""
from __future__ import division, print_function

import sys
import logging
from logging import debug, info, error
import numbers

import numpy as np
import scipy.optimize
import scipy.linalg
from numpy.fft import rfft, irfft, fft, ifft

from astropy.io import fits
import psrchive

class TemplateMatchError(ValueError):
    pass

def rotate_phase(prof, phase):
    """Rotate phase of profile earlier"""
    fprof = fft(prof)
    # top coefficient is special in even-length real FFTs
    # save it so phase shifting is non-destructive
    #topco = fprof[-1]
    n = len(fprof)
    fprof[:n//2] *= np.exp(2.j*np.pi*np.arange(n//2)*phase)
    fprof[-(n//2)+1:] *= np.exp(2.j*np.pi*np.arange(-(n//2)+1,0)*phase)
    #fprof[-1] = topco
    return ifft(fprof).real

def convert_template(template, nbins):
    """Convert template to match prof (in length)

    Either down- or up-sample template so that it is the same length
    as prof. Use the Fourier domain; either drop or pad with zero
    the extra Fourier coefficients.
    """
    return irfft(rfft(template),nbins)*(float(nbins)/len(template))

def align_profile(template, prof):
    """Use cross-correlation to align template optimally with prof

    Return phase so that prof is approximately equal to
    rotate_phase(template,phase)*amp + bg
    (actually this should be a least-squares minimization).

    Note that swapping template and prof simply changes the sign of
    the resulting phase; the code is otherwise symmetrical.

    The code requires the template to have the same length as the
    profile.

    FIXME: can fail if the shift is exactly a half-bin.
    """
    ftemplate = fft(template)
    fprof = fft(prof)
    fcorr = ftemplate*np.conj(fprof)
    fcorr[0] = 0 # Ignore the constant
    fcorr[len(fcorr)//2] = 0 # Ignore the annoying middle component
    corr = ifft(fcorr)
    i = np.argmax(np.abs(corr))
    iphase = float(i)/len(corr)
    n = len(fcorr)
    def peak(p):
        return -np.abs(np.sum(fcorr[1:n//2]
                                  *np.exp(2.j*np.pi*np.arange(1,n//2)*p))
                 +np.sum(fcorr[-(n//2)+1:]
                             *np.exp(2.j*np.pi*np.arange(-(n//2)+1,0)*p)))
    r = scipy.optimize.minimize_scalar(peak,
                    bracket=(iphase-2./len(corr),
                             iphase,
                             iphase+2./len(corr)))
    phase = (r.x+0.5)%1-0.5
    return phase

def align_scale_profile(template, prof):
    """Use cross-correlation to align template optimally with prof

    Return phase, amp, bg so that prof is approximately equal to
    rotate_phase(template,phase)*amp + bg
    (actually this should be a least-squares minimization).
    """
    phase = align_profile(template, prof)
    rtemp = rotate_phase(template,phase)
    tz = rtemp - np.mean(rtemp)
    pz = prof - np.mean(prof)
    amp = np.dot(tz, pz)/np.dot(tz,tz)
    bg = np.mean(prof)-np.mean(rtemp)*amp
    return phase, amp, bg

def rotate_phase_iquv(prof, phase):
    return np.array([rotate_phase(p, phase) for p in prof])

def pack_iquv(iquv):
    n, l = iquv.shape
    if n!=4:
        raise ValueError
    return np.reshape(iquv, l*n)
def unpack_iquv(d):
    ln = len(d)
    if ln % 4:
        raise ValueError
    l = ln//4
    return np.reshape(d, (4,l))

def mueller_fit_matrix(data, const=True, intensity_only=False):
    Acols = []
    names = []
    if intensity_only:
        M = np.zeros((4,4), dtype=data.dtype)
        M[0,0] = 1
        d = np.dot(M,data)
        Acols.append(pack_iquv(d))
        names.append(("M",(0,0)))
    else:
        for i in range(4):
            for j in range(4):
                M = np.zeros((4,4), dtype=data.dtype)
                M[i,j] = 1
                d = np.dot(M,data)
                Acols.append(pack_iquv(d))
                names.append(("M",(i,j)))
    if const:
        for i in range(4):
            d = np.zeros_like(data)
            d[i,:] = 1
            Acols.append(pack_iquv(d))
            names.append(("const",i))
    
    A = np.array(Acols).T
    return A, names
def mueller_from_fit(x, names):
    M = np.zeros((4,4), dtype=x.dtype)
    c = np.zeros(4, dtype=x.dtype)
    for (t,v),xi in zip(names,x):
        if t=="M":
            i,j = v
            M[i,j] = xi
        elif t=="const":
            c[v] = xi
        else:
            raise ValueError("Mysterious fit parameter %s" % ((t,v),))
    return M, c
    
def fit_mueller(goal, data, const=True, intensity_only=False):
    A, names = mueller_fit_matrix(data, const, intensity_only)
    x, rk, res, s = np.linalg.lstsq(A, pack_iquv(goal))
    d = np.dot(A, x)
    M, c = mueller_from_fit(x, names)
    return unpack_iquv(d), M, c

def wrap(x,center=0.):
    """Wrap x modulo 1 to the range [-0.5,0.5)"""
    return (x+0.5-center)%1 - 0.5+center

class MatchResult(object):
    """Results of template matching

    This is just a class to collect named data about the results of a
    template matching operation. The fields can be listed with dir(),
    but some key ones are described here:

    phase : float
        The phase determined by the template matching operation. 
        Typically this should be in the range [-0.5,0.5), but it is
        only meaningful modulo 1, and may occasionally leave this
        range.

    uncert : float
        The standard deviation of the phase attribute, if the data is
        accurately described by the template plus the claimed amount of
        noise. 

    uncert_scaled : float
        An estimate of the standard deviation obtained by using the 
        scatter of the residuals to estimate the per-bin noise.

    uncert_robust : float
        An estimate of the standard deviation based on "robust standard
        errors" (specifically the Mackinnon and White estimator), which
        tries to estimate uncertainties in the presence of 
        heteroskedasticity, that is, allowing for the possibility that
        each bin has a different amount of noise, to be estimated from
        the residuals.

    template_in_data_space : ndarray
        The template transformed by the best-fit transformation
        to match the data as well as possible.

    residuals : ndarray
        The data minus the best-fit template. This is the quantity
        whose squared size is minimized by the transformation described.

    data_in_template_space : ndarray
        The data, transformed by the inverse of the best-fit
        transformation.

    residuals_template : ndarray
        The template minus the data transformed to match it.
    """
    pass
def align_profile_polarization(template, data, 
                               global_search="phase", 
                               noise="off-pulse",
                               off_pulse_fraction=0.25,
                               intensity_only=False,
                               extra_outputs=[]):
    """Align template and data, allowing arbitrary polarization transformations
    
    If global_search is set, polarization fits are tried for a grid of
    phases before starting the local optimization. If it is not set,
    relatively fast cross-correlation of the intensity profile is used
    to obtain an initial guess for the local optimization. This shortcut
    may settle on a wrong local minimum if the input data calibration is
    bad enough, or if there are a sufficient profusion of local minima.
    
    The noise parameter allows the user to supply an estimate of the noise
    level in the profile, if available; if not the noise will be estimated
    from the fit residuals. 
    
    This function returns the best-fit phase, an estimate of the uncertainty
    on that phase, and a version of the template as closely matched to
    the data as possible, in terms of phase and polarization.
    """
    
    # FIXME: allow intensity-only templates, data, or fitting in any combination
    p, n = template.shape
    if p!=4:
        raise ValueError("Template should be 4 by n but is %d by %d" % (p,n))
    pd, nd = data.shape
    if pd!=4:
        raise ValueError("Data should be 4 by n but is %d by %d" % (pd,nd))
    if n != nd:
        template = np.array([convert_template(t, nd) for t in template])
        p, n = template.shape
    if isinstance(global_search, numbers.Number):
        phase = global_search
    elif global_search=='fourier':
        # scipy.linalg.lstsq allows solving for many RHS's at once
        # but we're trying to fit transformed, rotated versions of
        # the template to the data, and the fit matrix depends on
        # the template. So instead fit transformed versions of the
        # template to shifted versions of the data, then reverse the
        # shift.

        # Fitting in the Fourier domain should avoid many inverse FFTs
        # but getting it to agree exactly with the time-domain profile
        # comparison appears to be nontrivial

        phases = np.linspace(-0.5,0.5,2*n,endpoint=False)
        f_template = rfft(template,axis=-1)
        f_data = rfft(data,axis=-1)

        # give Nyquist the right weight (rfft)
        f_template[:,-1] /= np.sqrt(2)
        f_data[:,-1] /= np.sqrt(2)

        # make constants go away with indexing
        b = (f_data[:,1:,None]
             *np.exp(-2.j*np.pi*np.arange(1,f_data.shape[1])[None,:,None]
                     *phases[None,None,:]))
        # Don't rotate Nyquist
        b[:,-1,:] = f_data[:,-1,None]
        b = b.reshape((-1,len(phases))) # pack_iquv is a reshape
        A, names = mueller_fit_matrix(f_template[:,1:], const=False)
        x, res, rk, s = scipy.linalg.lstsq(A,b)
        phase = phases[np.argmin(res)]
    elif global_search:
        # scipy.linalg.lstsq allows solving for many RHS's at once
        # but we're trying to fit transformed, rotated versions of
        # the template to the data, and the fit matrix depends on
        # the template. So instead fit transformed versions of the
        # template to shifted versions of the data, then reverse the
        # shift.

        phases = np.linspace(-0.5,0.5,2*n,endpoint=False)
        b = np.array([pack_iquv(rotate_phase_iquv(data, -p)) for p in phases]).T
        A, names = mueller_fit_matrix(template)
        x, res, rk, s = scipy.linalg.lstsq(A,b)
        phase = phases[np.argmin(res)]
        #qofs = [qof(template, data, p) for p in phases]
        #phase = phases[np.argmin(qofs)]
    else:
        raise NotImplementedError

    def qof(template, data, phase):
        t_shift = rotate_phase_iquv(template,phase)
        t_fit, x, names = fit_mueller(data, t_shift, intensity_only=intensity_only)
        return np.sum((data-t_fit)**2)
    bracket = phase - 1/(2.*n), phase, phase + 1/(2.*n)
    try:
        r = scipy.optimize.minimize_scalar(lambda p: qof(template, data, p), bracket=bracket)
    except ValueError as err:
        if isinstance(global_search, numbers.Number):
            raise TemplateMatchError("Provided initial guess not close enough to fit")
        error(global_search)
        i = np.argmin(res)
        error(res[(i-1)%len(phases)], res[i], res[(i+1)%len(phases)])
        error(qof(template, data, bracket[0]), 
              qof(template, data, bracket[1]), 
              qof(template, data, bracket[2]))
        raise
    t_data = rotate_phase_iquv(template, r.x)
    t_data, M, c = fit_mueller(data, t_data, intensity_only=intensity_only)
    result = MatchResult()
    result.phase = r.x
    result.template_in_data_space = t_data
    result.residuals = data-t_data
    data_t = np.dot(scipy.linalg.pinv(M),
                    rotate_phase_iquv(data, -r.x)-c[:,None])
    result.residuals_template = template-data_t
    result.data_in_template_space = data_t
    result.M = M
    result.c = c
    
    noise_per_bin = None

    # Getting uncertainties
    A, names = mueller_fit_matrix(t_data, intensity_only=intensity_only)
    # FIXME: check sign
    d_shift = irfft(rfft(t_data,axis=-1)
                    *2.j*np.pi*np.arange(t_data.shape[1]//2+1),axis=-1)
    names.append("shift")
    A = np.hstack((A,pack_iquv(d_shift)[:,None]))
    # Now A is the least-squares fit matrix, at least locally.
    # The columns are related to the variables as described in names
    # In particular the last one is phase
    cov = scipy.linalg.pinv(np.dot(A.T, A))
    n_fit = 16+4+1
    bias_corr = A.shape[0]/(A.shape[0]-n_fit)

    # Base noise on off-pulse regions
    # Judge off-pulse-ness based on template total intensity
    # So don't use template in data space, since total intensity
    # might be mangled by polarization calibration
    r_pol = rotate_phase_iquv(template, result.phase)
    thresh = np.percentile(r_pol[0],100*off_pulse_fraction)
    c = r_pol[0]<=thresh
    res = np.ma.array(result.residuals)
    res[:,~c] = np.ma.masked
    noises = res.std(axis=-1).filled(np.nan)
    result.noises = noises

    if isinstance(noise, numbers.Number):
        result.noise_mode = "specified"
        noise_per_bin = noise
    elif noise=="off-pulse":
        result.noise_mode = noise
        # FIXME: deal with intensity-only data
        noise_per_bin = noises.mean()
    elif noise=="residuals":
        result.noise_mode = noise
        noise_per_bin = np.std(result.residuals)
    result.noise_per_bin = noise_per_bin

    if noise_per_bin is not None:
        result.snr = np.sqrt(n)*np.mean(np.std(result.template_in_data_space,axis=1))/noise_per_bin
        result.uncert = np.sqrt(cov[-1,-1])*noise_per_bin
        result.reduced_chi2 = bias_corr*np.mean(result.residuals**2)/noise_per_bin**2
    result.snr_residuals = np.sqrt(n)*np.mean(np.std(result.template_in_data_space,axis=1))/np.std(result.residuals)
    cov_scaled = cov*bias_corr*np.mean(result.residuals**2)
    result.uncert_scaled = np.sqrt(cov_scaled[-1,-1])
    if ("leverage" in extra_outputs
        or "cov_robust" in extra_outputs):
        res = pack_iquv(result.residuals)
        leverage = np.diag(np.dot(A,np.dot(cov,A.T)))
        result.leverage = unpack_iquv(leverage)
    if "effect_on_phase" in extra_outputs:
        result.effect_on_phase = unpack_iquv(np.dot(cov,A.T)[-1,:])*noise_per_bin
    if "uncert_robust" in extra_outputs:
        sf = (res/(1-leverage))**2
        cov_robust = np.dot(np.dot(cov,
                                   np.dot(A.T,A*sf[:,None])),
                            cov)
        result.uncert_robust = np.sqrt(cov_robust[-1,-1])
    return result

def plot_iquv(iquv,linestyle='none',marker='.',markersize=1):
    import matplotlib.pyplot as plt
    npol, nbin = iquv.shape
    t_phases = np.linspace(0,1,nbin,endpoint=False)
    plt.plot(t_phases, iquv[0], color='k', 
             linestyle=linestyle, marker=marker, markersize=markersize)
    plt.plot(t_phases, np.hypot(iquv[1],iquv[2]), color='r', 
             linestyle=linestyle, marker=marker, markersize=markersize)
    plt.plot(t_phases, iquv[3], color='b', 
             linestyle=linestyle, marker=marker, markersize=markersize)
    plt.xlabel("phase")

tel_codes = {
    "Arecibo": "ao",
    "wsrt": "wsrt",
    "i": "wsrt",
    "GBT": "gbt",
}

def generate_toa_info(template, filename, noise="off-pulse", off_pulse_fraction=0.25):
    F_fits = fits.open(filename)
    F = psrchive.Archive_load(filename)
    F.convert_state("Stokes")
    
    data = F.get_data()
    weights = F.get_weights()
    telescope = F.get_telescope()
    tel_code = tel_codes[telescope]

    nchan = F.get_nchan()
    bw = F.get_bandwidth()
    cf = F.get_centre_frequency()
    freqs = F_fits['SUBINT'].data['DAT_FREQ']
    if nchan==1 and len(freqs.shape)==1:
        # Aargh. FITS simplifies arrays.
        freqs = freqs[:,None]
    if freqs.shape != (len(F), nchan):
        raise ValueError("frequency array has shape %s instead of %s"
                         % (freqs.shape, (len(F),nchan)))
    for i in range(len(F)):
        debug("subint %d of %d",i,len(F))
        I = F.get_Integration(i)
        e = I.get_epoch()
        e_mjdi = e.intday()
        e_mjdf = np.longdouble(e.fracday())
        P = I.get_folding_period()
        for j in range(nchan):
            if weights[i,j]==0:
                continue
            debug("chan %d of %d",j,nchan)
            sub_data = data[i,:,j,:]
            r = align_profile_polarization(template, sub_data, noise=noise, off_pulse_fraction=off_pulse_fraction)
            # FIXME: check sign
            # FiXME: do we use doppler here or in tempo?
            dt = wrap(r.phase)*P/86400.
            mjdi, mjdf = e_mjdi, e_mjdf-dt
            mjdi, mjdf = mjdi + np.floor(mjdf), mjdf - np.floor(mjdf)
            assert 0<=mjdf<1
            mjd_string = "%d.%s" % (mjdi, ("%.20f" % mjdf)[2:])
            mjd = mjdi+np.longdouble(mjdf)
            assert np.abs(np.longdouble(mjd_string)-mjd)<1e-3/86400.
            uncert = r.uncert*P*1e6 # in us
            flags = dict(subint=i, chan=j, snr=r.snr,
                         reduced_chi2=r.reduced_chi2,
                         phase=r.phase, uncert=r.uncert,
                         uncert_scaled=r.uncert_scaled,
                         P=P, weighted_frequency=I.weighted_frequency(j),
                         bw=bw/nchan, tsubint=I.get_duration(),
                         nbin=sub_data.shape[1], 
                         )
            d = dict(mjd_string=mjd_string,
                     mjd=mjd,
                     file=filename,
                     freq=freqs[i,j],
                     tel=tel_code,
                     uncert=uncert,
                     flags=flags)
            yield d

def write_toa_info(F, toa_info):
    t = toa_info.copy()
    flagpart = " ".join("-"+k+" "+str(v) for k,v in t["flags"].items())
    t["flagpart"] = flagpart
    l = ("{file} {freq} {mjd_string} {uncert} {tel} "
         "{flagpart}").format(**t)
    F.write(l)
    F.write("\n")

def load_template(filename, realign=False):
    T = psrchive.Archive_load(filename)
    T.fscrunch()
    T.tscrunch()
    T.convert_state('Stokes')
    T.remove_baseline()
    t_pol = T.get_data()[0,:,0,:]
    if realign:
        a = np.angle(np.fft.fft(t_values)[1])/(2*np.pi)
        t_pol = rotate_phase_iquv(t_pol, -a)
    t_pol /= np.amax(t_pol)
    return t_pol

if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Compute pulse arrival times.")
    parser.add_argument("-t", "--template", required=True)
    parser.add_argument("files", nargs="+")
    parser.add_argument("--realign", 
                        help="realign template so the fundamental has phase zero",
                        action="store_true")
    args = parser.parse_args()

    t_pol = load_template(args.template, realign=args.realign)

    sys.stdout.write("FORMAT 1\n")
    for f in args.files:
        for t in generate_toa_info(t_pol, f):
            write_toa_info(sys.stdout, t)

