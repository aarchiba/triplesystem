#!/usr/bin/env python
from __future__ import division, print_function

import sys
import os
import subprocess
from glob import glob
import pickle
import shutil
import shlex
import numpy as np
from backports import tempfile
from os.path import join
import re
import time
from numpy.fft import rfft, irfft, fft, ifft
import scipy.optimize
import signal
import warnings

import astropy.units as u
import astropy.coordinates
from astropy.io import fits

from logging import info, debug, error, warning, critical

import psrchive
import residuals

import template_match
from template_match import rotate_phase, convert_template, align_profile, align_scale_profile

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

class ProcessingError(ValueError):
    pass

# Gathering raw data

data_location_permanent = "/psr_archive/hessels/archibald/0337+17"
data_location = "/data/archibald/0337+1715"
scratch_location = "/data/archibald/scratch"
par_db = EphemerisCollection(directory=join(data_location,"ephemerides"))
wsrt_raw_location = join(data_location_permanent,"raw","WSRT")
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
                if i!=0:
                    F = psrchive.Archive_load(f)
                    extra_info["nsubint"] += len(F)
                    if F.get_nchan()>extra_info["nchan"]:
                        # Some files must be missing a GPU
                        info("Later files have more channels")
                        extra_info["nchan"] = F.get_nchan()
                        extra_info["centre_frequency"] = F.get_centre_frequency()
                        extra_info["bw"] = F.get_bandwidth()
                    del F
                        
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

def copy_if_newer(fi, fo):
    if (not os.path.exists(fo)
            or os.path.getmtime(fo)<os.path.getmtime(fi)):
        shutil.copy(fi,fo)

def defaraday(meta, inpat, outpat, rm, flip_rm_sign=False):
    work_dir = meta["work_dir"]
    infiles = sorted(glob(join(work_dir,inpat+"_*.ar")))
    outfiles = [join(work_dir,outpat+"_%04d.ar"%i) for i in range(len(infiles))]
    for (fi, fo) in zip(infiles, outfiles):
        meta["RM"] = rm
        meta["flip_rm_sign"] = flip_rm_sign
        if flip_rm_sign:
            rm = -rm
        pam(fi, output=fo, R=rm)

def realign(meta, inpat, outpat, align_mode):
    work_dir = meta["work_dir"]
    infiles = sorted(glob(join(work_dir,inpat+"_*.ar")))
    outfiles = [join(work_dir,outpat+"_%04d.ar"%i) for i in range(len(infiles))]
    if "smearing" not in meta:
        meta["smearing"] = [None]*len(infiles)
    for i, (fi, fo) in enumerate(zip(infiles, outfiles)):
        if align_mode == "none":
            # NOTE: convert to psrfits to be sure the backend correction doesn't crash
            pam(fi, output=fo, a="psrfits")
        elif align_mode == "par":
            par = par_db.get_par_for(meta["mjd"])
            pam(fi, output=fo, E=par, a="psrfits", ephver="tempo")
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
                fio = re.sub(r"\.ar$",".calib", fi)
                # Find out whether a full Reception model is available by trying
                subprocess.check_call(["pac", "-TaS", "-d", cal_db, fi])
                if not os.path.exists(fio):
                    # Doesn't exist, use default
                    subprocess.check_call(["pac", "-Ta", "-d", cal_db, fi])
                if os.path.exists(fio+"P"):
                    shutil.copy(fio+"P", fo)
                    meta["calibration_type"] = "polarization"
                else:
                    shutil.copy(fio, fo)
                    meta["calibration_type"] = "flux"
            else:
                shutil.copy(fi,fo)
                meta["calibration_type"] = "none"
            subprocess.check_call(["./update_be_delay",fo])

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
        # Manual zapping
        g = glob(join(work_dir, "*_%04d.ar.paz" % i))
        if g:
            fname, = g
            zapchans, zapsubs = read_manual_zap(fname,nchan=meta["nchan"])
            if zapchans:
                zo.extend(("-z", " ".join(str(c) for c in zapchans)))
            if zapsubs:
                zo.extend(("-w", " ".join(str(s) for s in zapsubs)))
        if median_r is None:
            # FIXME: what happens if no manual or always zapping either?
            paz(fi, *zo, output=ft1)
        else:
            paz(fi, *zo, output=ft1, r=None, R=median_r)
        if meta["tel"] in ["AO"]:
            F = psrchive.Archive_load(ft1)
            # from obsys.dat
            arecibo_location = astropy.coordinates.EarthLocation(
                x=2390490.0*u.m,y=-5564764.0*u.m,z=1994727.0*u.m)
            radeg, decdeg = (F.get_coordinates().angle1.getDegrees(), 
                             F.get_coordinates().angle2.getDegrees()) 
            srcpos = astropy.coordinates.SkyCoord(ra=radeg*u.degree, 
                                                  dec=decdeg*u.degree, 
                                                  frame='icrs')
            za_res = []
            zas = []
            for i in range(F.get_nsubint()):
                I = F.get_Integration(i)
                # Use astropy to compute zenith angle; weirdly this is off by about
                # 0.01 degrees but sometimes the ZA values in the file are completely
                # bogus
                st = I.get_start_time().in_days()
                t = astropy.time.Time(st, format='mjd')
                altaz = srcpos.transform_to(astropy.coordinates.AltAz(
                    obstime=t, location=arecibo_location))
                za = 90-altaz.alt.value
                if za < 1.13:
                    # in practice, 1.13 seems to be where the telescope goes off source
                    # Though I've seen it go down to 1.06 in reality, the file I analyzed
                    # cut off at 1.13 degrees.
                    za_res.append(i)
                    zas.append(za)
            del F
            if za_res:
                info("Zapping subintegrations in ZA keyhole: %s", za_res)
                info("ZAs: %s", zas)
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
    if len(infiles)==0:
        raise ValueError("Input files mysteriously missing for pattern %s: %s" % (inpat,glob(join(work_dir,"*")),))
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

def read_manual_zap(fname,nchan=None):
    zapchans = []
    zapsubs = []
    for l in open(fname).readlines():
        s = shlex.split(l)
        for (i,k) in enumerate(s):
            if k=="-z":
                zapchans += [int(c) for c in s[i+1].split()]
            elif k=="-w":
                zapsubs += [int(c) for c in s[i+1].split()]
            elif k=="-Z":
                b,e = [int(c) for c in s[i+1].split()]
                zapchans += range(b,e+1)
            elif k=="-W":
                b,e = [int(c) for c in s[i+1].split()]
                zapsubs += range(b,e+1)
            elif k in ["-f","-F","-x","-X","-E","-s","-S"]:
                raise ValueError("Edit '%s' not supported" % k)
    if nchan is not None:
        n = len(zapchans)
        zapchans = [c for c in zapchans if c<nchan]
        if len(zapchans)!=n:
            error("bogus channel detected in zap instructions, deleting and continuing")
    if not zapchans and not zapsubs:
        raise ProcessingError("No zaps extracted from %s" % fname)
    zapchans.sort()
    zapsubs.sort()
    return zapchans, zapsubs

def process_observation(obs_dir, result_name,
                        work_dir=None,
                        align_mode="par",
                        cal_db="data/cal/cal.db",
                        median_r=20,
                        toa_bw=np.inf,
                        toa_time=10.,
                        rm=None,
                        flip_rm_sign=False):
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

        if not meta["raw_files"]:
            raise ProcessingError("No raw files listed in %s" % obs_dir)

        for i,f in enumerate(meta["raw_files"]):
            rn = join(work_dir, "raw_%04d.ar" % i)
            wf = join(obs_dir, f)
            if (not os.path.exists(rn)
                or os.path.getmtime(rn)<os.path.getmtime(wf)):
                shutil.copy(wf,rn)
        for i,f in enumerate(meta["raw_files"]):
            # Make copies of manual zapping files
            rn = join(work_dir, "raw_%04d.ar.paz" % i)
            wf = join(obs_dir, f+".paz")
            if (os.path.exists(wf)
                and (not os.path.exists(rn)
                     or os.path.getmtime(rn)<os.path.getmtime(wf))):
                shutil.copy(wf,rn)
        # Mark files by mode
        mode = "fold"
        if meta["tel"] == "GBT": 
            # we only have search-mode files from the GBT
            # They aren't marked as having been folded after the fact
            # The way to tell is by looking at the sample time "tbin"
            # This matters because there's a few-hundred-microsecond
            # difference in pulse arrival times
            F = fits.open(join(work_dir, "raw_0000.ar"))
            if F['subint'].header['tbin'] > 5e-6:
                mode = "search"
            del F
        meta["mode"] = mode

        realign(meta, "raw", "align",
                align_mode=align_mode)
        if rm is None:
            calibrate(meta, "align", "cal", cal_db)
        else:
            calibrate(meta, "align", "calint", cal_db)
            defaraday(meta, "calint", "cal",
                      rm=rm, flip_rm_sign=flip_rm_sign)
        # Check for extra zap list, e.g. pazi output
        # To produce this use pazi and hit p, which prints out an
        # equivalent paz command; the user would then have to save
        # this somehow; I recommend appending it to [filename].pazcmd
        # Probably best to apply this to an unscrunched file that has
        # been autozapped, so weight_*.ar. The metadata should keep
        # track of manual zapping so it can be plotted on the summary.
        # The code should support multiple paz commands.
        # FIXME: reweight WSRT data
        zap_rfi(meta, "cal", "zap", median_r=median_r)
        #reweight(meta, "zap", "weight")
        scrunch(meta, "zap", "scrunch",
                toa_bw=toa_bw, toa_time=toa_time)
        with open(join(work_dir,"process.pickle"),"wb") as f:
            pickle.dump(meta,f)

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

def prepare_toa_info(summary, match="pat", snr_plot_threshold=10.):
    meta = summary["meta"]
    observation = meta["observation"]
    processing_name = meta["processing_name"]
    template = meta["template_path"]
    sf, = sorted(glob(join(observation,processing_name,"scrunch_*.ar")))

    if match=="pat":
        pat_output = subprocess.check_output(["pat",
                                              "-s", template,
                                              "-f", "tempo2 IPTA",
                                              sf])
    elif match=="mueller":
        pat_output = subprocess.check_output(["python", "template_match.py",
                                              "-t", template,
                                              sf])
    else:
        raise ValueError("Invalid matching mode %s" % match)
        
    topo_toa = []
    toa_info = []
    for l in pat_output.split("\n"):
        if not l or l.startswith("FORMAT"):
            continue
        ls = l.split()
        mjd = float(ls[2])
        #print mjd, l
        d = dict(mjd_string=ls[2],
                 mjd=mjd,
                 file=ls[0],
                 freq=float(ls[1]),
                 uncert=float(ls[3]),
                 tel=ls[4],
                 flags=dict())
        for k, v in zip(ls[5::2],ls[6::2]):
            if not k.startswith("-"):
                raise ValueError("Mystery flag: %s %s" % (k,v))
            d["flags"][k[1:]] = v
        if (len(ls)-5) % 2:
            raise ValueError("Apparently improper number of flags: %d in %s"
                                 % (len(ls),ls))
        toa_info.append(d)
    summary["toa_info"] = toa_info
    meta["ntoa"] = len(toa_info)
    if len(toa_info)==0:
        raise ProcessingError("No TOAs generated for %s" % observation)

    with tempfile.TemporaryDirectory("triple") as td:
        tim = join(td,"toas.tim")
        with open(tim,"wt") as f:
            f.write("FORMAT 1\n")
            for t in toa_info:
                if "snr" in t["flags"] and float(t["flags"]["snr"])<snr_plot_threshold:
                    continue
                topo_toa.append(t["mjd"])
                template_match.write_toa_info(f, t)
        par = par_db.get_par_for(meta["mjd"])
        subprocess.check_call(["tempo", "-f", par, tim],
                              cwd=td)
        try:
            resid2 = residuals.read_residuals(join(td,"resid2.tmp"))
            meta["tempo_failed"] = False
            meta["rms_residual"] = 1e6*np.std(resid2.prefit_sec)
            meta["mean_residual_uncertainty"] = 1e6*np.mean(resid2.uncertainty)
            mr = np.average(resid2.prefit_sec, weights=1/resid2.uncertainty)
            meta["reduced_chi2"] = np.sqrt(np.mean(((resid2.prefit_sec-mr)
                                                    /resid2.uncertainty)**2))
            summary["prefit_sec"] = resid2.prefit_sec
            summary["uncertainty"] = resid2.uncertainty
            summary["bary_freq"] = resid2.bary_freq
            assert len(topo_toa)==len(resid2.prefit_sec)
            summary["topo_toa"] = np.array(topo_toa)
        except IOError as e:
            ti = [t for t in toa_info
                  if "snr" not in t["flags"] or float(t["flags"]["snr"])>=snr_plot_threshold]
            meta["tempo_failed"] = True
            meta["rms_residual"] = np.nan
            meta["mean_residual_uncertainty"] = np.mean(
                [t["uncert"] for t in ti])
            meta["reduced_chi2"] = np.nan
            summary["prefit_sec"] = np.zeros(len(ti))
            summary["uncertainty"] = 1e-6*np.array(
                [t["uncert"] for t in ti])
            summary["bary_freq"] = np.array(
                [t["freq"] for t in ti])
            summary["topo_toa"] = np.array(
                [t["mjd"] for t in ti])


def prepare_scrunched(summary):
    meta = summary["meta"]
    observation = meta["observation"]
    processing_name = meta["processing_name"]
    template = meta["template_path"]
    toa_info = summary["toa_info"]
    p_dir = join(observation,processing_name)
    summary["p_dir"] = p_dir
    nb = meta["nbin"]
    scrunched = sorted(glob(join(p_dir, "scrunch_*.ar")))

    T = psrchive.Archive_load(template)
    T.dedisperse()
    T.pscrunch()
    T.remove_baseline()
    t_values = convert_template(T.get_data()[0,0,0,:], meta["nbin"])
    t_phases = np.linspace(0,1,len(t_values),endpoint=False)

    snr_sum = 0
    snr_weight = 0
    snr_data = []

    min_f = meta["centre_frequency"]-meta["bw"]/2
    max_f = meta["centre_frequency"]+meta["bw"]/2
    if min_f>max_f:
        min_f, max_f = max_f, min_f
    summary["min_f"] = min_f
    summary["max_f"] = max_f

    for s in scrunched:
        F = psrchive.Archive_load(s)
        F.pscrunch()
        F.dedisperse()
        F.remove_baseline()
        # subint, chan, bin
        d = F.get_data()[:,0,:,:]
        w = F.get_weights()
        ns, nc, nb = d.shape
        snrs = np.ma.zeros((ns,nc))
        toa_by_index = {}
        subix = []
        for i in range(len(F)):
            I = F.get_Integration(i)
            subix.append(I.get_start_time().in_days())
        subix.append(I.get_end_time().in_days())
        subix = np.array(subix)
        chix = np.linspace(min_f, max_f, nc+1)
        for t in toa_info:
            if "chan" in t["flags"]:
                j = int(t["flags"]["chan"])
            else:
                j = np.searchsorted(chix,t["freq"])-1
                if meta["bw"] < 0:
                    j = nc-1-j
            if "subint" in t["flags"]:
                i = int(t["flags"]["subint"])
            else:
                i = np.searchsorted(subix,t["mjd"])-1
            #print(t["mjd"], t["freq"], i, j, min_f, max_f)
            if (i,j) in toa_by_index:
                raise ProcessingError("Problem matching TOAs with "
                                      "subintegrations for %s: (%d,%d duplicated)" 
                                      % (t,i,j))
            toa_by_index[i,j] = t
        for i in range(ns):
            for j in range(nc):
                if w[i,j] == 0:
                    snrs[i,j] = np.ma.masked
                    continue
                prof_ = d[i,j]
                phase, amp, bg = align_scale_profile(t_values, prof_)
                t_fit = rotate_phase(t_values, phase)*amp + bg
                snrs[i,j] = (np.sqrt(nb)
                                *np.std(t_fit)/np.std(prof_-t_fit))
                if (i,j) in toa_by_index:
                    if "snr" not in toa_by_index[i,j]["flags"]:
                        toa_by_index[i,j]["flags"]["snr"] = snrs[i,j]
                else:
                    raise ProcessingError("Missing TOA for unzapped "
                                         "subint (%d,%d)" % (i,j))
        ts = 0
        te = (F.end_time()-F.start_time()).in_days()*86400
        snr_data.append((ts,te,snrs))
        snr_sum += snrs.sum()
        snr_weight += snrs.count()
    meta["average_snr"] = snr_sum/snr_weight
    summary["snr_data"] = snr_data
    summary["snr_sum"] = snr_sum
    summary["snr_weight"] = snr_weight
    summary["te"] = te
    summary["t_values"] = t_values

def accumulate(a, weights, axis=None):
    s = (a*weights).sum(axis=axis)
    w = weights.sum(axis=axis)
    a = np.ma.array(s.copy())
    a[w==0] = np.ma.masked
    a /= w
    return a, s, w

def prepare_unscrunched(summary):
    meta = summary["meta"]
    observation = meta["observation"]
    processing_name = meta["processing_name"]
    template = meta["template_path"]
    p_dir = summary["p_dir"]
    t_values = summary["t_values"]
    nb = meta["nbin"]
    unscrunched = sorted(glob(join(p_dir, "zap_*.ar")))

    gtp_data = None
    gtp_sum = None
    gtp_weight = None
    prof_data = None
    prof_sum = None
    prof_weight = None
    std_data = None
    std_sum = None
    std_weight = None

    yfp_data = []
    yfp_start_end = []

    smear_data = []

    for (i,u) in enumerate(unscrunched):
        F = psrchive.Archive_load(u)
        F.convert_state('Stokes')
        F.dedisperse()
        F.remove_baseline()
        # axes are (subint, polarization, channel, bin)
        d = F.get_data()
        # axes are (subint, channel)
        w = F.get_weights()
        if F.get_nchan() != meta["nchan"]:
            raise ProcessingError("File %s has %d channels but expected %d"
                                  % (u, F.get_nchan(), meta["nchan"]))

        sm = meta["smearing"][i].copy()
        if "raw_smearing" in meta:
            sm += meta["raw_smearing"][i]
        sm_xs = np.linspace((F.start_time().in_days()-meta["tstart"])*86400,
                            (F.end_time().in_days()-meta["tstart"])*86400,
                            len(F)+1)[1:-1]
        smear_data.append((sm_xs,sm))
        if "max_smearing" not in meta:
            meta["max_smearing"] = 0
        try:
            meta["max_smearing"] = max(meta["max_smearing"],
                                       np.amax(np.abs(sm))*1e6*meta["P"])
        except ValueError:
            error("Strange problem computing max_smearing; sm is %s", sm)

        # Profile
        # FIXME: this accumulation interacts badly with masked arrays
        # Specifically, if ever it becomes all masked, new nonzero weights won't
        # fix the problem. The solution is to work with sums instead, and divide
        # by the weights afterward.
        sa, sd, sw = accumulate(d, weights=w[:,None,:,None]+0*d,
                                axis=2)
        sa, sd, sw = accumulate(sd, weights=sw, axis=0)
        if prof_data is None:
            #print("Initializing profile with %g (%g) total weight from %s" % (np.sum(sw),np.sum(w), u))
            prof_data, prof_sum, prof_weights = sa, sd, sw
        else:
            #print("Adding %g (%g) total weight to profile from %s" % (np.sum(sw), np.sum(w), u))
            prof_sum += sd
            prof_weights += sw
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prof_data = prof_sum/prof_weights
            prof_data = np.ma.array(prof_data)
            prof_data[prof_weights==0] = np.ma.masked
            # weights should only be zero if entire observation was zapped
        #import matplotlib.pyplot as plt
        #plt.figure()
        #template_match.plot_iquv(prof_sum, linestyle="-")

        # Noise std. dev per bin (pre-averaging)
        sd = np.std(d[:,0,:,:], axis=-1)
        sa, sd, sw = accumulate(sd,weights=w)
        if std_data is None:
            std_data, std_sum, std_weight = sa, sd, sw
        else:
            std_sum += sd
            std_weight += sw
            std_data = std_sum/std_weight

        # GTp plot
        sa, sd, sw = accumulate(d[:,0], weights=w[...,None]+0*d[:,0],
                                axis=0)
        if gtp_data is None:
            gtp_data, gtp_sum, gtp_weights = sa, sd, sw
        else:
            gtp_sum += sd
            gtp_weights += sw
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gtp_data = gtp_sum/gtp_weights
            gtp_data = np.ma.array(gtp_data)
            gtp_data[gtp_weights==0] = np.ma.masked

        # YFp plot
        ya, yd, yw = accumulate(d[:,0], weights=w[...,None]+0*d[:,0],
                                axis=1)
        b = (F.start_time().in_days()-meta["tstart"])*86400
        e = (F.end_time().in_days()-meta["tstart"])*86400
        info("total weight from %s is %g" % (u, yw.sum()))
        yfp_data.append(ya)
        yfp_start_end.append((b,e))
    gtp_data = np.ma.array(gtp_data)
    gtp_data[gtp_weights==0] = np.ma.masked

    if np.sum(prof_weights)==0:
        raise ProcessingError("All data appears to have been zapped")
    #if prof_data.count()==0:
    #    raise ProcessingError("The entire profile is masked but weights are nonzero")
    phase, amp, bg = align_scale_profile(t_values, prof_data[0])
    t_fit = rotate_phase(t_values, phase)*amp + bg
    osnr = float(np.sqrt(nb)*np.std(t_fit)/np.std(prof_data[0]-t_fit))
    if np.isnan(osnr):
        raise ProcessingError("Problem with SNR computation; profile is %s, weights are %s" % (prof_data, prof_weights))
    meta["overall_snr"] = osnr
    summary["gtp_data"] = gtp_data
    summary["gtp_weights"] = gtp_weights
    summary["yfp_data"] = yfp_data
    summary["yfp_start_end"] = yfp_start_end
    summary["smear_data"] = smear_data
    summary["prof_data"] = prof_data
    summary["prof_weights"] = prof_weights
    summary["t_fit"] = t_fit

def plot_summary(summary):
    import matplotlib.pyplot as plt
    min_f = summary["min_f"]
    max_f = summary["max_f"]
    yfp_data = summary["yfp_data"]
    gtp_data = summary["gtp_data"]
    topo_toa = summary["topo_toa"]
    smear_data = summary["smear_data"]
    p_dir = summary["p_dir"]
    uncertainty = summary["uncertainty"]
    snr_sum = summary["snr_sum"]
    snr_data = summary["snr_data"]
    toa_info = summary["toa_info"]
    snr_weight = summary["snr_weight"]
    text_format = summary["text_format"]
    meta = summary["meta"]
    bary_freq = summary["bary_freq"]
    te = summary["te"]
    prefit_sec = summary["prefit_sec"]
    yfp_start_end = summary["yfp_start_end"]
    prof_data = summary["prof_data"]
    prof_weights = summary["prof_weights"]
    t_fit = summary["t_fit"]

    fig = plt.figure()
    fig.set_size_inches(10,8)

    lm = 0.085
    prof = plt.axes((lm,0.70,0.4,0.25))
    resid = plt.axes((lm,0.55,0.4,0.15))
    gtp = plt.axes((lm,0.30,0.4,0.25))
    yfp = plt.axes((lm,0.05,0.4,0.25))
    text_x, text_y = 0.50, 0.95

    cbar = plt.axes((0.80,0.70,0.15,0.03))
    snr = plt.axes((0.55,0.55,0.4,0.15))
    smear = plt.axes((0.55,0.40,0.4,0.15))
    resid_t = plt.axes((0.55,0.25,0.4,0.15))
    resid_f = plt.axes((0.55,0.05,0.4,0.15))

    tend = (meta["tend"]-meta["tstart"])*86400

    rcolor = "black"
    if meta["tempo_failed"]:
        rcolor = "gray"
    plt.sca(resid_t)
    plt.errorbar((topo_toa-meta["tstart"])*86400,
        1e6*prefit_sec,
        1e6*uncertainty,
        linestyle="none", fmt="k.", color=rcolor)
    plt.xlabel("t (s)")
    plt.ylabel("Residual ($\mu$s)")

    plt.sca(resid_f)
    plt.errorbar(bary_freq,
        1e6*prefit_sec,
        1e6*uncertainty,
        linestyle="none", fmt="k.", color=rcolor)
    plt.xlabel("f (MHz)")
    plt.ylabel("Residual ($\mu$s)")

    plt.sca(snr)
    for ts, te, snrs in snr_data:
        if meta["bw"]>0:
            snrs = snrs[:,::-1]
        plt.imshow(snrs.T, extent=(ts, te, min_f, max_f),
              interpolation='nearest')
    snr.set_aspect('auto')
    plt.ylabel("f (MHz)")
    plt.sca(resid_t)
    plt.xlim(0,tend)
    plt.sca(snr)
    plt.tick_params(axis='x', labelbottom='off')
    plt.xlim(0,tend)
    cb = plt.colorbar(cax=cbar, orientation='horizontal', label="S/N")
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    plt.sca(smear)
    plt.tick_params(axis='x', labelbottom='off')
    plt.xlim(0,tend)
    plt.ylabel("Smearing ($\mu$s)")

    plt.sca(yfp)
    #for (yd, (b,e)) in zip(yfp_data, yfp_start_end):
    #    plt.imshow(yd[::-1,:],extent=(0,1,b,e),
    #          interpolation='none')
    yd = np.ma.concatenate(yfp_data)
    b = yfp_start_end[0][0]
    e = yfp_start_end[-1][1]
    n, x = np.percentile(np.ma.compressed(yd),[1,99])
    plt.imshow(yd[::-1,:],extent=(0,1,b,e),
               vmin=n, vmax=x,
          interpolation='none')

    plt.sca(smear)
    for (sm_xs, sm) in smear_data:
        plt.plot(sm_xs, sm*1e6*meta["P"], color='k')

    plt.sca(prof)
    ps = np.linspace(0,1,prof_data.shape[1],endpoint=False)
    plt.plot(ps,prof_data[0,:], color='black')
    plt.plot(ps,prof_data[3,:], color='blue')
    plt.plot(ps,np.hypot(prof_data[1,:],prof_data[2,:]), color='red')
    plt.plot(ps, t_fit, color='green')
    plt.tick_params(axis='x', labelbottom='off')
    plt.ylabel("Flux density (mJy?)")
    p_x = ps[np.argmax(t_fit)]
    vline_alpha = 0.5
    vline_color = 'purple'
    plt.axvline(p_x, color=vline_color, alpha=vline_alpha)
    plt.xlim(0,1)

    plt.sca(resid)
    plt.plot(ps,prof_data[0,:]-t_fit, color='black')
    plt.ylabel(r"$\Delta$Flux")
    plt.tick_params(axis='x', labelbottom='off')
    plt.axvline(p_x, color=vline_color, alpha=vline_alpha)
    plt.xlim(0,1)

    plt.sca(gtp)
    if meta["bw"]<0:
        gtp_data_display = gtp_data[::-1,:]
    else:
        gtp_data_display = gtp_data
    n, x = np.percentile(np.ma.compressed(gtp_data),[1,99])
    plt.imshow(gtp_data_display[::-1,:], extent=(0,1,min_f,max_f),
               vmin=n, vmax=x,
              interpolation='none')
    gtp.set_aspect("auto")
    plt.ylabel("freq (MHz)")
    plt.tick_params(axis='x', labelbottom='off')
    plt.axvline(p_x, color=vline_color, alpha=vline_alpha)
    plt.xlim(0,1)

    plt.sca(yfp)
    yfp.set_aspect("auto")
    plt.xlabel("pulse phase")
    plt.ylabel("time (s)")
    plt.axvline(p_x, color=vline_color, alpha=vline_alpha)
    plt.xlim(0,1)
    plt.ylim(0,(meta["tend"]-meta["tstart"])*86400)

    plt.sca(resid_f)
    plt.xlim(min_f, max_f)

    text = text_format.format(**meta)

    plt.text(text_x, text_y, text,
             horizontalalignment="left", verticalalignment="top",
             transform=fig.transFigure)

    plt.viridis()


def make_toas(observation, processing_name, toa_name, template,
              summary_plot=True, match="pat"):
    # FIXME: take advantage of extra information available from mueller matrix fitting
    with open(join(observation,processing_name,"process.pickle"),"rb") as f:
        meta = pickle.load(f)
    summary = dict(meta=meta)
    meta["observation"] = observation
    meta["processing_name"] = processing_name
    meta["template"] = os.path.basename(template)
    meta["template_path"] = template
    meta["toa_name"] = toa_name
    text_format = """PSR J0337+1715 observation {name}
    Observed: {tel} {receiver} Processing: {processing_name}
    Template: {template} TOAs: {toa_name}
    Center frequency: {centre_frequency:.1f} MHz
    Length: {length:.1f} s Bandwidth: {bw:.1f} MHz
    Maximum smearing: {max_smearing:.2f} $\mu$s
    Signal-to-noise ratio overall: {overall_snr:.1f} Average: {average_snr:.1f}
    RMS residual: {rms_residual:.2f} $\mu$s # TOAs: {ntoa}
    Mean residual uncertainty: {mean_residual_uncertainty:.2f} $\mu$s
    Residual reduced $\chi^2$: {reduced_chi2:.2f}
    """
    summary["text_format"] = text_format
    prepare_toa_info(summary, match=match)
    prepare_scrunched(summary)
    p_dir = summary["p_dir"]
    toa_info = summary["toa_info"]
    prepare_unscrunched(summary)
    meta["summary"] = text_format.format(**meta)
    tdir = join(p_dir,meta["toa_name"])
    if not os.path.exists(tdir):
        os.makedirs(tdir)
    with open(join(tdir,"toas.tim"),"wt") as of:
        of.write("FORMAT 1\n")
        for t in toa_info:
            if "nch" in t["flags"]:
                del t["flags"]["nch"]
            for f in ["processing_name", "max_smearing",
                          "band", "tel", "toa_name"]:
                t["flags"][f] = meta[f]
            if "mode" in meta:
                # not worth reprocessing all data to add mode flag
                # users of new TOAs can presume unmarked data to be
                # fold mode.
                t["flags"]["mode"] = meta["mode"]

            flagpart = " ".join("-"+k+" "+str(v) for k,v in t["flags"].items())
            t["flagpart"] = flagpart
            l = ("{file} {freq} {mjd_string} {uncert} {tel} "
                     "{flagpart}").format(**t)
            of.write(l)
            of.write("\n")
    with open(join(tdir,"toas.pickle"),"wt") as of:
        meta["toas"] = toa_info
        pickle.dump(meta, of)
    with open(join(tdir,"summary.pickle"),"wt") as of:
        pickle.dump(summary, of)
    if summary_plot:
        import matplotlib.pyplot as plt
        plot_summary(summary)
        plt.savefig(join(tdir,"summary.pdf"))
