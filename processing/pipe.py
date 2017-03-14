#!/usr/bin/env python
from __future__ import division, print_function

import sys
import os
import subprocess
from glob import glob
import pickle
import shutil
import numpy as np
from backports import tempfile
from os.path import join
import re
import time
from numpy.fft import rfft, irfft, fft, ifft
import scipy.optimize

from logging import info, debug, error, warning, critical

import psrchive

# tools for make-like conditional rerunning

def ensure_list(l):
    if isinstance(l,basestring) or isinstance(l,int):
        l = [l]
    return l

def touch(f):
    if not os.path.exists(f):
        raise ValueError("File %s does not exist yet" % f)
    subprocess.check_call(["touch",f])

def need_rerun(inputs, outputs):
    """Examine inputs and outputs and return whether a command should be rerun.

    If one of the outputs does not exist, or if the modification date of the
    newest input is newer than the oldest output, return True; else False. The
    idea is to allow make-like behaviour.
    """
    inputs = ensure_list(inputs)
    outputs = ensure_list(outputs)

    if len(outputs)==0:
        raise ValueError("No outputs specified")

    io = inputs
    inputs = []
    for i in io:
        if i.startswith("@"):
            for l in open(i[1:]).readlines():
                inputs.append(l.strip())
        else:
            inputs.append(i)

    oldest_out = np.inf
    oldest_out_name = None

    for o in outputs:
        if not os.path.exists(o):
            info("Output %s missing" % o)
            return True
        ot = os.path.getmtime(o)
        if ot<oldest_out:
            oldest_out = ot
            oldest_out_name = o

    for i in inputs:
        if os.path.getmtime(i) > oldest_out:
            info("Input %s newer than %s" % (i,oldest_out_name))
            debug("%s > %s" %
                      (time.ctime(os.path.getmtime(i)),
                        time.ctime(os.path.getmtime(oldest_out_name))))
            return True

    return False

def write_file_if_changed(fname, s):
    """Write the string s to the file fname but only if it's different"""

    if not os.path.exists(fname) or open(fname,"rt").read() != s:
        with open(fname, "wt") as f:
            f.write(s)


class Command(object):
    """Run a command from the set of Fermi tools.

    Commands are automatically rerun only if necessary. Upon construction of
    the object, the input and output file arguments are listed; for keyword
    arguments, the name is given, while for positional arguments, the position.

    On calling this object, positional arguments appear in positions, and
    keyword arguments are appended in the form key=value. Two special keyword
    arguments are recognized:

    rerun determines whether to force a rerun of the command.
    True means always rerun, False means never, and None (the default)
    means rerun if necessary.

    call_id is a string describing this particular call. If provided,
    standard out and standard error are saved to files and can be displayed
    even if a rerun is not necessary. If not provided, they will be seen
    only if the command is actually run.
    """
    def __init__(self, command,
                     infiles=[], outfiles=[],
                     inplace={},
                     always_args=[]):
        self.command = ensure_list(command)
        self.infiles = ensure_list(infiles)
        self.outfiles = ensure_list(outfiles)
        self.always_args = ensure_list(always_args)
        if len(self.outfiles)==0:
            raise ValueError("No output files specified")
        self.inplace = dict(inplace)
        for (k,v) in self.inplace.items():
            if k not in self.infiles:
                raise ValueError("Parameter %s to modify inplace not listed"
                                     "among input parameters: %s"
                                     % (k,self.infiles))
            if v not in self.outfiles:
                raise ValueError("Destination parameter %s for inplace"
                                     "modification not listed among output"
                                     "parameters: %s"
                                     % (v,self.outfiles))

    def format_kwargs(self, kwargs):
        raise NotImplementedError

    def detect_failure(self, returncode, stdout, stderr):
        return returncode!=0

    def __call__(self, *args, **kwargs):
        args = list(args)
        rerun = kwargs.pop("rerun", None)
        call_id = kwargs.pop("call_id", None)

        infiles = []
        for k in self.infiles:
            try:
                infiles.append(args[k])
            except IndexError:
                infiles.append(kwargs[k])
        outfiles = []
        for k in self.outfiles:
            try:
                outfiles.append(args[k])
            except TypeError:
                outfiles.append(kwargs[k])
        stdout_name, stderr_name, args_name = [outfiles[0]+"."+s
                                                   for s in ["stdout",
                                                             "stderr",
                                                             "args"]]
        infiles.append(args_name)
        outfiles.extend((stderr_name, stdout_name))

        if os.path.exists(args_name):
            old_args = pickle.load(open(args_name,"r"))
        else:
            info("No old arguments on record")
            old_args = ([], {})
        new_args = (args, kwargs)
        if new_args != old_args:
            info("Arguments changed")
            debug("%s != %s" % (new_args, old_args))
            with open(args_name,"w") as f:
                pickle.dump(new_args, f)

        if rerun or (rerun is None and need_rerun(infiles, outfiles)):
            success = False
            try:
                if self.inplace:
                    for (k, v) in self.inplace.items():
                        try:
                            debug("Inplace replacing position %d=%s "
                                      "with keyword %s=%s",
                                      k, args[k], v, kwargs[v])
                            shutil.copy(args[k],kwargs[v])
                            args[k] = kwargs[v]
                        except TypeError:
                            debug("Inplace replacing keyword %s=%s "
                                      "with keyword %s=%s",
                                      k, kwargs[k], v, kwargs[v])
                            shutil.copy(kwargs[k],kwargs[v])
                            kwargs[k] = kwargs[v]
                        del kwargs[v]
                with open(stdout_name,"w") as stdout, \
                  open(stderr_name, "w") as stderr:
                    cmd = (self.command
                            +self.always_args
                            +[str(a) for a in args]
                            +self.format_kwargs(kwargs))
                    P = subprocess.Popen(cmd,
                                         stdout=stdout.fileno(),
                                         stderr=stderr.fileno())
                    try:
                        P.communicate()
                    except KeyboardInterrupt:
                        P.send_signal(signal.SIGINT)
                        raise
                stdout = open(stdout_name,"r").read()
                stderr = open(stderr_name,"r").read()
                if self.detect_failure(P.returncode, stdout, stderr):
                    raise ValueError("Command %s failed with return code %d.\n"
                                         "stdout:\n%s\n"
                                         "stderr:\n%s\n"
                                        % (" ".join(self.command
                                                        +list(args)
                                                        +self.format_kwargs(kwargs)),
                                               P.returncode,
                                               stdout,
                                               stderr))
                sys.stdout.write(stdout)
                sys.stderr.write(stderr)
                success = True
            finally:
                if not success:
                    for f in outfiles:
                        try:
                            os.unlink(f)
                        except OSError as e:
                            sys.stderr.write("Problem deleting %s: %s" % (f,e))
        else: # no need to rerun
            sys.stdout.write(open(stdout_name).read())
            sys.stderr.write(open(stderr_name).read())

class FermiCommand(Command):
    def format_kwargs(self, kwargs):
        fmtkwargs = []
        for (k,v) in kwargs.items():
            fmtkwargs.append("%s=%s" % (k,v))
        return fmtkwargs
class Tempo2Command(Command):
    def format_kwargs(self, kwargs):
        fmtkwargs = []
        for (k,v) in kwargs.items():
            fmtkwargs.append("-%s" % k)
            fmtkwargs.append(str(v))
        return fmtkwargs
    def detect_failure(self, returncode, stdout, stderr):
        debug("tempo2 returned %d" % returncode)
        # tempo2 return codes are useless
        if stderr.strip():
            return True
        return False

tempo2 = Tempo2Command("tempo2", infiles=["f", "ft1", "ft2"],
                           outfiles=["outfile"],
                           inplace={"ft1":"outfile"})

# FIXME: how to handle positional input arguments?
class PsrchiveCommand(Command):
    def format_kwargs(self, kwargs):
        fmtkwargs = []
        for (k,v) in kwargs.items():
            if len(k)==1:
                fmtkwargs.append("-"+k)
            else:
                fmtkwargs.append("--"+k)
            if v is not None:
                fmtkwargs.append(str(v))
        return fmtkwargs
pam = PsrchiveCommand("pam",
       infiles=[0],
       outfiles=["output"],
       inplace={0:"output"},
       always_args=["-m"])
paz = PsrchiveCommand("paz",
       infiles=[0],
       outfiles=["output"],
       inplace={0:"output"},
       always_args=["-m"])

class CalledProcessError(subprocess.CalledProcessError):
    def __init__(self, cmd, returncode, output=None, error=None):
        self.cmd = cmd
        self.returncode = returncode
        self.output = output
        self.error = error

    def __str__(self):
        return """CalledProcessError(
            cmd=%s,
            returncode=%d,
            output='''%s''',
            error='''%s''')""" % (self.cmd,
                                      self.returncode,
                                      self.output,
                                      self.error)

def check_call(cmd, stderr_re=None, shell=False, cwd=None):
    P = subprocess.Popen(cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=shell,
                    cwd=cwd)
    output, error = P.communicate()
    returncode = P.returncode
    del P
    if returncode or (stderr_re is not None and re.search(stderr_re, error)):
        raise CalledProcessError(cmd, returncode, output=output, error=error)




# working with observations

def WSRT_observations(base_dir):
    pass

def AO_observations(base_dir):
    pass

def GBT_observations(base_dir):
    pass



class Observation(object):

    def __init__(self, psrchive_file):
        pass

    def find_best_cal_file(self):
        pass

    def calibrate(self):
        pass

    def zap(self):
        pass

    def align(self, new_ephemeris, max_smear=None):
        pass

    def reweight(self):
        pass

    def downsample(self):
        pass

    def make_toas(self):
        pass


class EphemerisCollection(object):

    def __init__(self, directory, spacing=4, mjdbase=55920,
        fit_segment_dir="/misc/astron/archibald/projects/triplesystem/processing"):
        self.directory = directory
        self.spacing = spacing
        self.mjdbase = mjdbase
        self.fit_segment_dir = fit_segment_dir
        if not os.path.exists(self.directory):
            os.mkdir(directory)

    def _generate(self, mjd, name):
        with tempfile.TemporaryDirectory() as td:
            fs = self.fit_segment_dir + "/fit_segment.py"
            check_call(["python", fs,
                        "--toafile", self.fit_segment_dir + "/fake.tim",
                        "--pulsesfile", self.fit_segment_dir + "/fake.pulses",
                        "--length", str(2*self.spacing),
                        str(mjd)],
                       cwd=td)
            shutil.copy(td+"/J0337+17.par", name)

    def get_par_for(self, mjd):
        par_mjd = int(self.mjdbase
                    +self.spacing*np.round((mjd-self.mjdbase)/self.spacing))
        name = self.directory + "/%s.par" % par_mjd
        if not os.path.exists(name):
            self._generate(par_mjd, name)
        return name




def rotate_phase(prof, phase):
    """Rotate phase of profile earlier"""
    fprof = fft(prof)
    #assert len(fprof)!=len(prof), "Must use numpy-style rfft not scipy-style"
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


# Metadata contents:
# name - observation name
# input_files - list of files
# tstart - MJD of observation start
# tend - MJD of observation end
# mjd - MJD of observation middle
# nsubint - number of subintegrations in observation
# nchan - number of channels in observation (must be the same in all files)
# tel - GBT, AO, or WSRT
# receiver - lbw, ...
# raw_files - list of basenames of "raw" files (for WSRT these have been
#      aligned with the longterm ephemeris and combined); input to processing
# P - pulsar nominal spin period
# bw - bandwidth in MHz
# centre_frequency - centre frequency in MHz
# band - 350, 430, 1400, 2000, ?
# raw_smearing - list of arrays of between-subint smearing; only for WSRT data
#      realigned during assembly
# smearing - list of arrays of between-subint smearing after realignment
# manual_zap_subints - list of lists
# manual_zap_channels - list of lists
# calibration_type - none, flux, or polarization

# Gathering raw data

data_location = "/psr_archive/hessels/archibald/0337+17"
scratch_location = "/data/archibald/scratch"
par_db = EphemerisCollection(directory=join(data_location,"ephemerides"))
wsrt_raw_location = join(data_location,"raw","WSRT")
wsrt_obs_glob = join(wsrt_raw_location,"*","*0337*")
longterm_par = ("/misc/astron/archibald/projects/"
    "triplesystem/processing/longterm.par")

name_template = "{mjd:.2f}_{tel}_{band}"

class DiscontinuityError(ValueError):
    pass

def smearing(F,G):
    P = F.get_Integration(0).get_folding_period()
    deltas = []
    for i in range(len(F)):
        d = (G.get_Integration(i).get_epoch()
                 -F.get_Integration(i).get_epoch()).in_days()*86400
        deltas.append(d/P)
    deltas = np.array(deltas)

    smear = np.diff(deltas)
    smear = (smear+0.5) % 1 - 0.5

    return smear

def clear_wsrt_day(d, success=False, failure=True):
    success_file = join(d,"success.pickle")
    failure_file = join(d,"failure.pickle")
    if success and os.path.exists(success_file):
        os.unlink(success_file)
    if failure and os.path.exists(failure_file):
        os.unlink(failure_file)

def band(center_frequency, tel):
    if 1000<center_frequency<1800:
        band = 1400
    elif 1800<=center_frequency<2500:
        band = 2000
    elif 2500<=center_frequency<3500:
        band = 3000
    elif 700<center_frequency<900:
        band = 800
    elif 400<=center_frequency<500:
        band = 430
    elif 300<center_frequency<400:
        band = 350
    else:
        raise ValueError("Unknown band: %f MHz at %s" % (center_frequency, tel))
    return band

def tel_name(t):
    if t == 'i':
        return 'WSRT'
    elif t == 'GBT':
        return 'GBT'
    elif t == 'Arecibo':
        return 'AO'
    else:
        raise ValueError("Unrecognized get_telescope result: %s" % t)

def receiver_name(F):
    tel = tel_name(F.get_telescope())
    if tel == "WSRT":
        return "PuMa2_"+str(band(F.get_centre_frequency(), tel))
    else:
        rcvr = F.get_receiver_name()
        if rcvr == "L-wide":
            rcvr = "lbw"
        return rcvr

def divisors(n):
    """Return a sorted list of the divisors of n"""
    divs = []
    i=1
    while i**2<n:
        j, r = divmod(n,i)
        if r==0:
            divs.append(i)
            divs.append(j)
        i += 1
    if i**2==n:
        divs.append(i)
    divs.sort()
    return divs

def process_wsrt_day(d, work_dir=None):
    """Process a day's worth of WSRT observations

    Goes through all the individual files and bands taken on a particular day
    and tries to assemble a single archive from all of them. In order for this
    to work it is necessary to apply an updated ephemeris. This is because
    psradd -R needs to reapply the ephemeris to each subband, and some of the
    old ephemerides are no longer understood (in particular, pre-bugfix BTX).
    In addition to the single output file, a dictionary of relevant information
    is recorded, including in particular the maximum smearing within any
    subintegration. The output is stored in a single directory.

    Success or failure are recorded in the input directory, and if either
    has already been recorded, the saved result is returned instead of
    reprocessing.

    Arguments
    ---------

    d : string
        The directory containing the observation; typically of the form
        .../11Apr2012/J0337+1715-1380
    work_dir : string
        The directory to be used for temporary files; if not provided a
        temporary directory will be created and then deleted when done.

    Returns
    -------

    extra_info : dict
        A copy of the metadata dictionary that is saved to disk.

    """
    if not os.path.isdir(d):
        info("Not a directory: %s", d)
        return
    info("Processing WSRT directory %s", d)
    success_file = join(d,"success.pickle")
    failure_file = join(d,"failure.pickle")
    if os.path.exists(success_file):
        with open(success_file, "rb") as f:
            return pickle.load(f)
    if os.path.exists(failure_file):
        with open(failure_file, "rb") as f:
            raise pickle.load(f)
    with tempfile.TemporaryDirectory(prefix="triple",
                                     dir=scratch_location) as td:
        if work_dir is not None:
            if os.path.exists(work_dir):
                shutil.rmtree(work_dir)
            os.makedirs(work_dir)
            td = work_dir
        try:
            extra_info = dict(input_files=[],
                              tel="WSRT",
                              raw_smearing=[])
            for s in sorted(glob(join(d,"Band*"))):
                files = sorted(glob(join(s,"u*.ar")))
                if not files:
                    continue
                extra_info["input_files"].extend(files)
                b = os.path.basename(s)
                tadd = join(td,b+".tadd.ar")
                check_call(["psradd", "-o", tadd]+files)
                F = psrchive.Archive_load(tadd)

                tstart = F.get_Integration(0).get_start_time().in_days()
                tend = F.get_Integration(len(F)-1).get_end_time().in_days()
                mjd = (tstart+tend)/2
                #par = par_db.get_par_for(mjd)
                par = longterm_par
                check_call(["pam",
                            "--ephver", "tempo",
                            "-E", par,
                            "-e", "align.ar",
                            tadd])
                align = join(td,b+".tadd.align.ar")
                G = psrchive.Archive_load(align)

                if len(extra_info["raw_smearing"])==0:
                    extra_info["raw_smearing"].append(smearing(F,G))
                extra_info["tstart"] = tstart
                extra_info["tend"] = tend
                extra_info["mjd"] = mjd

                del F
                del G

            check_call(["psradd",
                            "-o", join(td,"raw.ar"),
                            "-R"]+glob(join(td,"Band*.tadd.align.ar")))
            F = psrchive.Archive_load(join(td,"raw.ar"))
            l = (F.end_time()-F.start_time()).in_seconds()
            T = F.integration_length()
            extra_info["length"] = T
            extra_info["nsubint"] = len(F)
            extra_info["nchan"] = F.get_nchan()
            extra_info["centre_frequency"] = F.get_centre_frequency()
            extra_info["P"] = F.get_Integration(0).get_folding_period()
            extra_info["receiver"] = receiver_name(F)
            extra_info["bw"] = F.get_bandwidth()
            extra_info["nbin"] = F.get_nbin()
            extra_info["band"] = band(extra_info["centre_frequency"],
                                          extra_info["tel"])
            extra_info["name"] = name_template.format(**extra_info)
            extra_info["obs_dir"] = join(data_location,
                                             "obs",extra_info["name"])
            extra_info["raw_files"] = [join(extra_info["obs_dir"],
                                                  "raw_0000.ar")]
            del F
            if np.abs(l-T)>5:
                raise DiscontinuityError(
                    "Observation in directory %s appears not to be contiguous: "
                    "total integration time %f but start-to-end time %f"
                    % (d,T,l))
            if not os.path.exists(extra_info["obs_dir"]):
                os.makedirs(extra_info["obs_dir"])
            shutil.copy(join(td,"raw.ar"),
                        extra_info["raw_files"][0])
            with open(success_file,"wb") as f:
                pickle.dump(extra_info, f)
            with open(join(extra_info["obs_dir"],"meta.pickle"),"wb") as f:
                pickle.dump(extra_info, f)
            return extra_info
        except Exception as e:
            error("failure processing %s: %s", d, e)
            with open(failure_file, "wb") as f:
                pickle.dump(e,f)
            raise

def process_uppi_batch(d,mjd,obsid,work_dir=None):
    info("Processing uppi batch %s %s in %s", mjd, obsid, d)
    files = sorted(glob(join(d,"*_"+mjd+"_*_"+obsid+"_*.fits")))
    if not files: # no file number part maybe? should leave just one
        pat = join(d,"*_"+mjd+"_*_"+obsid)
        #info("trying without file number: %s", pat)
        files = sorted(glob(pat))
    if not files:
        raise ValueError("No files found")
    success_file = files[0]+".success"
    failure_file = files[0]+".failure"
    if os.path.exists(success_file):
        with open(success_file, "rb") as f:
            return pickle.load(f)
    if os.path.exists(failure_file):
        with open(failure_file, "rb") as f:
            raise pickle.load(f)
    with tempfile.TemporaryDirectory(prefix="triple",
                                     dir=scratch_location) as td:
        if work_dir is not None:
            if os.path.exists(work_dir):
                shutil.rmtree(work_dir)
            os.makedirs(work_dir)
            td = work_dir
        try:
            F = psrchive.Archive_load(files[0])
            if F.get_nsubint() == 0:
                raise ValueError("No integrations in %s" % (files[0]))
            del F

            extra_info = dict(input_files=files,
                              max_smearing=0,
                              nsubint=0)
            tfs = []
            for (i,f) in enumerate(files):
                tf = join(td,"raw_%04d.fits" % (i+1))
                shutil.copy(f, tf)
                tfs.append(tf)
            F = psrchive.Archive_load(tfs[0])
            st = F.get_Integration(0).get_start_time().in_days()
            F = psrchive.Archive_load(tfs[-1])
            et = F.get_Integration(len(F)-1).get_end_time().in_days()
            mjd = (st+et)/2

            extra_info["mjd"] = mjd
            extra_info["tstart"] = st
            extra_info["tend"] = et
            extra_info["tel"] = tel_name(F.get_telescope())

            T = 86400*(et-st)
            extra_info["length"] = T
            extra_info["centre_frequency"] = F.get_centre_frequency()
            extra_info["P"] = F.get_Integration(0).get_folding_period()
            extra_info["receiver"] = receiver_name(F)
            extra_info["bw"] = F.get_bandwidth()
            extra_info["nbin"] = F.get_nbin()
            extra_info["nsubint"] += len(F)
            extra_info["nchan"] = F.get_nchan()
            extra_info["band"] = band(extra_info["centre_frequency"],
                                          extra_info["tel"])
            extra_info["name"] = name_template.format(**extra_info)
            extra_info["obs_dir"] = join(data_location,
                                             "obs",extra_info["name"])
            extra_info["raw_files"] = []
            del F

            if not os.path.exists(extra_info["obs_dir"]):
                os.makedirs(extra_info["obs_dir"])
            for (i,f) in enumerate(tfs):
                tf = "raw_%04d.ar" % i
                shutil.copy(f,
                    join(extra_info["obs_dir"], tf))
                extra_info["raw_files"].append(tf)
            with open(success_file,"wb") as f:
                pickle.dump(extra_info, f)
            with open(join(extra_info["obs_dir"],"meta.pickle"),"wb") as f:
                pickle.dump(extra_info, f)
            return extra_info

        except Exception as e:
            error("failure processing %s: %s", d, e)
            with open(failure_file, "wb") as f:
                pickle.dump(e,f)
            raise


def process_uppi_dir(d):
    ids = set()
    for f in glob(join(d,"*.fits")):
        fs = os.path.basename(f).split("_")
        if len(fs) == 5:
            uppi, mjd, _, obsid, _ = fs
        elif len(fs) == 4:
            uppi, mjd, _, obsid = fs
        else:
            continue
        if uppi not in ["puppi", "guppi", "GUPPI"]:
            continue
        ids.add((mjd,obsid))
    for (mjd,obsid) in sorted(ids):
        try:
            process_uppi_batch(d,mjd,obsid)
        except ValueError as e:
            error("Problem with %s %s %s: %s", d, mjd, obsid, e)
        except RuntimeError as e:
            error("Weird problem with %s %s %s: %s", d, mjd, obsid, e)

# Standard zap commands (paz arguments) to apply per receiver
# 'default' is applied to any reciever not listed
zap = {
        'Rcvr1_2':["-F","1100 1150",
                   "-F","1250 1262",
                   "-F","1288 1300",
                   "-F","1373 1381",
                   "-F","1442 1447",
                   "-F","1525 1558",
                   "-F","1575 1577",
                   "-F","1615 1630",
                   "-F","1370 1385"],
        'Rcvr_800':["-F","794.6 798.6",
                    "-F","814.1 820.7"],
        '327':[],
        '430':["-F","380 420",
               "-F","446 480"],
        'lbw':["-F","980 1150",
                  "-F","1618 1630"],
        'sbw':["-F","1600 1770",
                  "-F","1880 2050",
                  "-F","2100 2160",
                  "-F","2400 2600"],
        'PuMa2_350':[],
        'PuMa2_1400':[],
    }

def cleanup(obs, work_dir=None):
    with open(join(obs,"meta.pickle"),"rb") as f:
        M = pickle.load(f)
    if "clean_files" in M:
        all = True
        for f in M["clean_files"]:
            if not os.path.exists(join(obs,f)):
                all = False
                break
        if all:
            return M
    if "align_files" not in M: # forgotten on first-pass WSRT
        M["align_files"] = ["align.ar"]

    with tempfile.TemporaryDirectory(prefix="triple",
                                     dir=scratch_location) as td:
        if work_dir is not None:
            if os.path.exists(work_dir):
                shutil.rmtree(work_dir)
            os.makedirs(work_dir)
            td = work_dir
        tfs = [join(td,os.path.basename(f)) for f in M["align_files"]]
        for (f,tf) in zip(M["align_files"], tfs):
            shutil.copy(join(obs,f),tf)
        r = re.compile(r"\.[^.]*$")
        cfs = [r.sub(".calib", tf) for tf in tfs]
        cfPs = [r.sub(".calibP", tf) for tf in tfs]
        zfs = [r.sub(".zap", tf) for tf in tfs]
        wfs = [r.sub(".wt", tf) for tf in tfs]
        if M["tel"] == "WSRT":
            for tf, cfP in zip(tfs, cfPs):
                shutil.copy(tf, cfP)
        else:
            # FIXME: don't hardcode the path
            check_call(["pac","-Ta",
                            "-d","data/cal/cal.db"]
                           + tfs)
        if "receiver" not in M: # forgotten on first pass
            M['receiver'] = receiver_name(psrchive.Archive_load(tfs[0]))
        zo = zap[M['receiver']]
        check_call(["paz", "-r", "-R", "20", "-e", "zap"]
                       + zo + cfPs)
        if M["tel"] in ["AO", "GBT"]:
            for zf in zfs:
                F = psrchive.Archive_load(zf)
                za_res = []
                for i in range(F.get_nsubint()):
                    I = F.get_Integration(i)
                    if I.get_telescope_zenith() < 1.5:
                        za_res.append(i)
                if za_res:
                    info("Zapping subintegrations in keyhole: %s", za_res)
                    check_call(["paz", "-m", "-w",
                                " ".join([str(i) for i in za_res]),
                                zf])
        # Make RFI plot?

        M['clean_files'] = ["clean_%04d.ar" % i
                                for i in range(len(M["align_files"]))]
        M['nsubint'] = 0
        M['zaps'] = None
        for cf, f in zip(M['clean_files'], zfs):
            F = psrchive.Archive_load(f)
            w = F.get_weights()
            M['nsubint'] += F.get_nsubint()
            M['nchan'] = F.get_nchan()
            if M['zaps'] is None:
                M['zaps'] = [0 for i in range(M['nchan'])]
            for i in range(M['nchan']):
                M['zaps'][i] += np.sum(w[:,i]==0)

            shutil.copy(f, join(obs,cf))

        with open(join(obs,"meta.pickle"),"wb") as f:
            pickle.dump(M,f)

        return M

def copy_if_newer(fi, fo):
    if (not os.path.exists(fo)
            or os.path.getmtime(fo)<os.path.getmtime(fi)):
        shutil.copy(fi,fo)

def realign(meta, inpat, outpat, align_mode):
    work_dir = meta["work_dir"]
    infiles = sorted(glob(join(work_dir,inpat+"_*.ar")))
    outfiles = [join(work_dir,outpat+"_%04d.ar"%i) for i in range(len(infiles))]
    if "smearing" not in meta:
        meta["smearing"] = [None]*len(infiles)
    for i, (fi, fo) in enumerate(zip(infiles, outfiles)):
        if align_mode == "none":
            copy_if_newer(fi,fo)
        elif align_mode == "par":
            par = par_db.get_par_for(meta["mjd"])
            pam(fi, output=fo, E=par, ephver="tempo")
            F = psrchive.Archive_load(fi)
            G = psrchive.Archive_load(fo)
            meta["smearing"][i] = smearing(F,G)
            del F
            del G
        elif align_mode == "polyco":
            tstart, tend = meta["tstart"], meta["tend"]
            raise NotImplementedError
        else:
            raise ValueError("align_mode '%s' not recognized" % align_mode)

def calibrate(meta, inpat, outpat, cal_db):
    work_dir = meta["work_dir"]
    infiles = sorted(glob(join(work_dir,inpat+"_*.ar")))
    outfiles = [join(work_dir,outpat+"_%04d.ar"%i) for i in range(len(infiles))]
    for fi, fo in zip(infiles, outfiles):
        if (not os.path.exists(fo)
            or os.path.getmtime(fo)<os.path.getmtime(fi)):
            if cal_db is not None and meta["tel"] != "WSRT":
                subprocess.check_call(["pac", "-Ta", "-d", cal_db, fi])
                fio = re.sub(r"\.ar$",".calib", fi)
                if os.path.exists(fio+"P"):
                    shutil.copy(fio+"P", fo)
                    meta["calibration_type"] = "polarization"
                else:
                    shutil.copy(fio, fo)
                    meta["calibration_type"] = "flux"
            else:
                shutil.copy(fi,fo)
                meta["calibration_type"] = "none"

def zap_rfi(meta, inpat, outpat, median_r):
    work_dir = meta["work_dir"]
    infiles = sorted(glob(join(work_dir,inpat+"_*.ar")))
    outfiles = [join(work_dir,outpat+"_%04d.ar"%i) for i in range(len(infiles))]
    info("Zapping %s", infiles)
    for i, (fi, fo) in enumerate(zip(infiles, outfiles)):
        ft1 = join(work_dir, "zaptemp_%04d.ar" % i)
        ft2 = join(work_dir, "zaptemp2_%04d.ar" % i)
        # zap always-bad channels
        zo = zap[meta['receiver']]
        paz(fi, *zo, output=ft1, r=None, R=median_r)
        # FIXME: allow manual zapping
        if meta["tel"] in ["AO", "GBT"]:
            F = psrchive.Archive_load(ft1)
            za_res = []
            for i in range(F.get_nsubint()):
                I = F.get_Integration(i)
                if I.get_telescope_zenith() < 1.5:
                    za_res.append(i)
            del F
            if za_res:
                info("Zapping subintegrations in keyhole: %s", za_res)
                paz(ft1, output=ft2, w=" ".join([str(i) for i in za_res]))
            else:
                copy_if_newer(ft1,ft2)
        else:
            copy_if_newer(ft1,ft2)
        if False: # Manual zapping
            pass
        else:
            copy_if_newer(ft2,fo)

def scrunch(meta, inpat, outpat, toa_bw, toa_time):
    work_dir = meta["work_dir"]
    infiles = sorted(glob(join(work_dir,inpat+"_*.ar")))
    outfiles = [join(work_dir,"scrunchtemp_%04d.ar"%i)
                    for i in range(len(infiles))]
    add_file = join(work_dir,"scrunchadd_0000.ar")
    out_file = join(work_dir,outpat+"_0000.ar")
    divs = divisors(meta["nchan"])
    # can only downsample by integer factors
    # pick the next larger downsampling factor
    i = np.searchsorted(divs, abs(meta["nchan"]*toa_bw/meta["bw"]))
    if i==len(divs):
        i -= 1
    nchan_out = meta["nchan"]//divs[i]
    t = -np.inf
    # fscrunch
    for i, (fi, fo) in enumerate(zip(infiles, outfiles)):
        pam(fi, output=fo, setnchn=nchan_out)
        t = max(t, os.path.getmtime(fo))
    # combine to single file
    if (not os.path.exists(add_file)
            or t>os.path.getmtime(add_file)):
        subprocess.check_call(["psradd","-o",add_file]+outfiles)
    # tscrunch
    F = psrchive.Archive_load(add_file)
    nsubint_add = len(F)
    nsub_out = min(max(1,np.ceil(meta["length"]/toa_time)),
                   nsubint_add)
    if nsubint_add != meta['nsubint']:
        error("Metadata claims %d subints but files contain %d "
                             "for observation %s; overriding"
                             % (meta['nsubint'], nsubint_add, meta['name']))
    meta['nsubint'] = nsubint_add
    pam(add_file, output=out_file, setnsub=nsub_out)
    F = psrchive.Archive_load(out_file)
    debug("Scrunch requested %d channels from %d got %d",
              nchan_out, meta["nchan"], F.get_nchan())
    debug("Scrunch requested %d subints from %d got %d",
              nsub_out, meta["nsubint"], F.get_nsubint())
    meta["scrunch_nchan"] = F.get_nchan()
    meta["scrunch_nsubint"] = F.get_nsubint()
    del F

def process_observation(obs_dir, result_name,
                        work_dir=None,
                        align_mode="par",
                        cal_db="data/cal/cal.db",
                        median_r=20,
                        toa_bw=np.inf,
                        toa_time=10.):
    """Process observation through to TOAs

    Arguments
    ---------

    obs_dir : string
        Directory the observation data is stored in
    work_dir : string or None
        Directory to carry the work out in; if None, create a
        temporary directory and then delete it when done
    align_mode : "none", "par", or "polyco"
        Specify how alignment is to be carried out; "none" means
        use the native phase alignment of the input file, "par"
        means to generate short-term par files, and "polyco"
        means to generate polycos for the observation
    cal_db : string or None
        Path to the calibration database used py pac;
        if None, don't calibrate
    median_r : integer
        Width to use in automatic median-filter RFI zapping
    toa_bw : float
        Bandwidth in MHz to attempt to use in TOA generation; the
        band will be divided into equal parts no wider than this
    toa_time : float
        Time in seconds to attempt to use in TOA generation;
        the observation will be divided into equal parts no longer
        than this
    """


    with tempfile.TemporaryDirectory(prefix="triple",
                                     dir=scratch_location) as temp_dir:
        if work_dir is None:
            work_dir = temp_dir
        elif not os.path.exists(work_dir):
            os.makedirs(work_dir)

        with open(join(obs_dir,"meta.pickle"),"rb") as f:
            meta = pickle.load(f)

        meta["work_dir"] = work_dir

        for i,f in enumerate(meta["raw_files"]):
            rn = join(work_dir, "raw_%04d.ar" % i)
            wf = join(obs_dir, f)
            if (not os.path.exists(rn)
                or os.path.getmtime(rn)<os.path.getmtime(wf)):
                shutil.copy(wf,rn)

        realign(meta, "raw", "align",
                align_mode=align_mode)
        calibrate(meta, "align", "cal", cal_db)
        # Check for extra zap list, e.g. pazi output
        # To produce this use pazi and hit p, which prints out an
        # equivalent paz command; the user would then have to save
        # this somehow; I recommend appending it to [filename].pazcmd
        # Probably best to apply this to an unscrunched file that has
        # been autozapped, so weight_*.ar. The metadata should keep
        # track of manual zapping so it can be plotted on the summary.
        # The code should support multiple paz commands.
        zap_rfi(meta, "cal", "zap",
            median_r=median_r)
        # FIXME: reweight WSRT data
        #reweight(meta, "zap", "weight")
        scrunch(meta, "zap", "scrunch",
                toa_bw=toa_bw, toa_time=toa_time)
        with open(join(work_dir,"process.pickle"),"wb") as f:
            pickle.dump(meta,f)
        # Summary plots are more useful after TOA generation
        #make_summary_plots(meta, "summary.pdf")

        if not os.path.exists(join(obs_dir,result_name)):
            os.makedirs(join(obs_dir,result_name))
        for p in ["zap_*.ar",
                  "scrunch_*.ar",
                  "*.pdf",
                  "process.pickle"]:
            for f in glob(join(work_dir,p)):
                copy_if_newer(f, join(obs_dir,result_name,os.path.basename(f)))

def generate_toa(obs_dir, processing_name, toa_name,
                 template,
                 template_method="correlation",
                 work_dir=None):
    with tempfile.TemporaryDirectory(prefix="triple",
                                     dir=scratch_location) as temp_dir:
        if work_dir is None:
            work_dir = temp_dir
        elif not os.path.exists(work_dir):
            os.makedirs(work_dir)

        p_dir = join(obs_dir, processing_name)
        with open(join(p_dir,"process.pickle"),"rb") as f:
            meta = pickle.load(f)

        for f in glob(join(p_dir,"scrunch_*.ar")):
            pat
