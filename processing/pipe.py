#!/usr/bin/env python

import os
import subprocess
from glob import glob
import pickle

from logging import info, debug, error, warning, critical

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
                     inplace={}):
        self.command = ensure_list(command)
        self.infiles = ensure_list(infiles)
        self.outfiles = ensure_list(outfiles)
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
        rerun = kwargs.pop("rerun", None)
        call_id = kwargs.pop("call_id", None)

        infiles = [kwargs[f] for f in self.infiles]
        outfiles = [kwargs[f] for f in self.outfiles]
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
                        shutil.copy(kwargs[k],kwargs[v])
                        kwargs[k] = kwargs[v]
                        del kwargs[v]
                with open(stdout_name,"w") as stdout, \
                  open(stderr_name, "w") as stderr:
                    cmd = (self.command
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
