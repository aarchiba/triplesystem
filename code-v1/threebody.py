import os
import string
import random
from glob import glob
import time
import sys
import logging
import cPickle as pickle
import inspect

import numpy as np
import numpy.random
import scipy.linalg
import scipy.optimize

import astropy.table
import astropy.time

import kepler
import quad_integrate

if True:
    logger = logging.getLogger(__name__)
    debug = logger.debug
    error = logger.error
else:
    def debug(s):
        pass

#              parfile = '0337_tempo2_nobinary.par',
def mjd_fromstring(s, base_mjd=0):
    i,f = s.split(".")
    i = int(i)-base_mjd
    f = np.float128("0."+f)
    return i+f

def read_t2_toas(fname):
    toa_info = []
    for l in open(fname).readlines():
        if not l or l.startswith("FORMAT"):
            continue
        ls = l.split()
        mjd = float(ls[2])

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
    return toa_info

def write_t2_toas(fname, toa_info):
    with open(fname,"wt") as F:
        F.write("FORMAT 1\n")
        for toa_info in toa_infos:
            flagpart = " ".join("-"+k+" "+str(v) for k,v in t["flags"].items())
            t["flagpart"] = flagpart
            l = ("{file} {freq} {mjd_string} {uncert} {tel} "
                 "{flagpart}").format(**t)
            F.write(l)
            F.write("\n")


abbreviate_infos = {
    'Arecibo,L-band,PUPPI_coherent_fold': 'AO1440',
    'GBT,L-band,GUPPI_coherent_fold': 'GBT1500',
    'GBT,L-band,GUPPI_coherent_search': 'GBT1500s',
    'WSRT,L-band,PUMAII_coherent_fold': 'WSRT1400',
    }

#              tempo2_program = '/home/aarchiba/software/tempo2/tempo2/tempo2',
#              tempo2_dir = '/home/aarchiba/software/tempo2/t2dir',
astro_names = ["d_RAJ","d_DECJ","d_PX","d_PMRA","d_PMDEC"]
def load_toas(timfile = '0337+17.tim',
              pulses = '0337+17.pulses',
              parfile = '0337_bogus.par',
              tempo2_program = 'tempo2',
              tempo2_dir = os.environ['TEMPO2'],
              t2outfile = '0337+17.out',
              base_mjd = None):
    """Load and barycenter pulse-numbered TOAs

    This function loads in TOAs from a .tim file, uses a .par file
    to apply the barycentric correction (Earth position and DM), and
    returns the TOAs and the pulse numbers that go with them. The
    output of tempo2 is cached and re-used if available, so if the
    input file is changed the output file should be deleted or its
    name should be changed.
    """
    if base_mjd is None:
        base_mjd = 0

    try:
        pulses = np.loadtxt(pulses,dtype=np.float128)
    except IOError as e:
        if e.errno != 2:
            raise
        pulses = None
    timfile_len = len(open(timfile,"rt").readlines())
    try:
        if t2outfile is None:
            t2outfile = ""
        o = open(t2outfile).read()
    except IOError:
        import subprocess, os
        e = os.environ.copy()
        e['TEMPO2'] = tempo2_dir
        outline = "OUTPUT {bat} {freq} {err} {%s}\n"%("} {".join(astro_names))
        o = subprocess.check_output([tempo2_program,
            "-nobs", str(timfile_len+100),
            "-npsr", "1",
            "-output", "general2", "-s", outline,
            "-f", parfile, timfile], env=e)
        if t2outfile:
            with open(t2outfile,"wt") as f:
                f.write(o)

    t2_bats = []
    freqs = []
    errs = []
    derivs = {n:[] for n in astro_names}
    for l in o.split("\n"):
        if not l.startswith("OUTPUT"):
            continue
        try:
            t2_bats.append(mjd_fromstring(l.split()[1], base_mjd))
            freqs.append(float(l.split()[2]))
            errs.append(float(l.split()[3]))
            for n,w in zip(astro_names,l.split()[4:]):
                derivs[n].append(np.float128(w))
        except ValueError:
            raise ValueError("Unable to decode line '%s'" % l)
    t2_bats = np.array(t2_bats)
    errs = np.array(errs)*1e-6 # convert to s
    for n in astro_names:
        derivs[n] = np.array(derivs[n])
    if pulses is not None and len(t2_bats)!=len(pulses):
        raise ValueError("Confusing tim file %s: list of %d BATs doesn't match list of %d pulse numbers" % (timfile,len(t2_bats),len(pulses)))
    telcodes = []
    for l in open(timfile).readlines():
        if l.split():
            telcode = l.split()[0]
            if telcode in ["i", "j", "1", "3", "f", "@"]:
                telcodes.append(telcode)
    if len(telcodes)!=len(t2_bats):
        raise ValueError("Confusing tim file %s: list of %d telescope codes doesn't match list of %d BATs" % (timfile,len(telcodes),len(t2_bats)))

    tels = []
    tel_list = []
    def pick_tel(f):
        if 348<f<352:
            tel = 'GBT350'
        elif 326<f<328:
            tel = 'AO327'
        elif 330<f<346:
            tel = 'WSRT350'
        elif 1357.8<f<1358.5 or 1449.8<f<1450.5:
            tel = 'AO1350'
        elif 1450.5<f<1600 or 1700<f<1800:
            tel = 'GBT1500'
        elif 1420<f<1440.1:
            tel = 'AO1440'
        elif 1330<f<1390:
            tel = 'WSRT1400'
        elif 810<f<830:
            tel = 'GBT820'
        elif 99999<f:
            tel = 'infinite'
        else:
            print "WARNING: unable to determine telescope for frequency '%s'" % f
            tel = 'mystery'
        return tel
    def pick_tel_code(f,c):
        tel = None
        if c=='i' or c=='j':
            if 1200<f<1500:
                tel = 'WSRT1400'
        elif c=='1':
            if 1000<f<1900:
                tel = 'GBT1500'
            elif 300<f<400:
                tel = 'GBT350'
            elif 700<f<900:
                tel = 'GBT820'
        elif c=='3':
            if 1357.8<f<1358.5 or 1449.8<f<1450.5:
                tel = 'AO1350'
            elif 1420<f<1440.1:
                tel = 'AO1440'
        elif c=='f':
            if 1000.0<f<2500:
                tel = 'NCY1400'
        elif c=='@':
            tel = 'infinite'

        if tel is None:
            print "WARNING: unable to determine telescope for frequency '%s' and code '%s'" % (f,c)
            tel = 'mystery'
        return tel

    for f,c in zip(freqs,telcodes):
        tel = pick_tel_code(f,c)
        if tel not in tel_list:
            tel_list.append(tel)
    tel_list.sort()
    for f,c in zip(freqs,telcodes):
        tel = pick_tel_code(f,c)
        tels.append(tel_list.index(tel))
    tels = np.array(tels)

    ix = np.argsort(t2_bats)
    for n in astro_names:
        if len(derivs[n])==0:
            continue
        derivs[n] = derivs[n][ix]
    if pulses is None:
        return t2_bats[ix], None, tel_list, tels[ix], errs[ix], derivs, ix
    else:
        return t2_bats[ix], pulses[ix], tel_list, tels[ix], errs[ix], derivs, ix

def load_pipeline_toas(timfile,
              parfile = '0337_bogus.par',
              tempo2_program = 'tempo2',
              tempo2_dir = os.environ['TEMPO2'],
              t2outfile = None,
              base_mjd = None):
    """Load and barycenter pulse-numbered TOAs

    This function loads in TOAs from a .tim file, uses a .par file
    to apply the barycentric correction (Earth position and DM), and
    returns the TOAs and the pulse numbers that go with them. The
    output of tempo2 is cached and re-used if available, so if the
    input file is changed the output file should be deleted or its
    name should be changed.
    """
    if base_mjd is None:
        base_mjd = 0

    toa_info = read_t2_toas(timfile)

    try:
        if t2outfile is None:
            t2outfile = ""
        o = open(t2outfile).read()
    except IOError:
        import subprocess, os
        e = os.environ.copy()
        e['TEMPO2'] = tempo2_dir
        outline = "OUTPUT {bat} {freq} {err} {%s}\n"%("} {".join(astro_names))
        o = subprocess.check_output([tempo2_program,
            "-nobs", str(len(toa_info)+100),
            "-npsr", "1",
            "-output", "general2", "-s", outline,
            "-f", parfile, timfile], env=e)
        if t2outfile:
            with open(t2outfile,"wt") as f:
                f.write(o)

    t2_bats = []
    freqs = []
    errs = []
    derivs = {n:[] for n in astro_names}
    for l in o.split("\n"):
        if not l.startswith("OUTPUT"):
            continue
        try:
            t2_bats.append(mjd_fromstring(l.split()[1], base_mjd))
            freqs.append(float(l.split()[2]))
            errs.append(float(l.split()[3]))
            for n,w in zip(astro_names,l.split()[4:]):
                derivs[n].append(np.float128(w))
        except ValueError:
            raise ValueError("Unable to decode line '%s'" % l)
    t2_bats = np.array(t2_bats)
    errs = np.array(errs)*1e-6 # convert to s
    for n in astro_names:
        derivs[n] = np.array(derivs[n])
    if len(t2_bats) != len(toa_info):
        raise ValueError("tempo2 produced %d outputs but we found %d TOAs" % (len(t2_bats), len(toa_info)))
    pulses = np.zeros(len(toa_info),dtype=np.int64)
    for i,t in enumerate(toa_info):
        if "pn" in t["flags"]:
            pulses[i] = np.int64(t["flags"])
    if np.any(pulses==0):
        pulses = None
    tel_list = []
    telcode_dict = { 
        ("WSRT",1400): "WSRT1400",
        ("AO",1400): "AO1350",
        ("GBT",1400): "GBT1500",
        ("WSRT",350): "WSRT350",
        ("AO",430): "AO430",
    }
    for t in toa_info:
        k = t["flags"]["tel"], int(t["flags"]["band"])
        telcode = telcode_dict[k]
        if telcode not in tel_list:
            tel_list.append(telcode)
    tel_list.sort()

    tels = []
    for t in toa_info:
        k = t["flags"]["tel"], int(t["flags"]["band"])
        telcode = telcode_dict[k]
        tels.append(tel_list.index(telcode))
    tels = np.array(tels)

    ix = np.argsort(t2_bats)
    for n in astro_names:
        if len(derivs[n])==0:
            continue
        derivs[n] = derivs[n][ix]
    if pulses is None:
        return t2_bats[ix], None, tel_list, tels[ix], errs[ix], derivs, ix
    else:
        return t2_bats[ix], pulses[ix], tel_list, tels[ix], errs[ix], derivs, ix



def trend_matrix(mjds, tel_list, tels,
    const=True, P=True, Pdot=True, jumps=True,
    position=False, proper_motion=False, parallax=False,
    derivs=None,
    f0 = 365.9533436144258189, pepoch = 56100, mjdbase = 55920,
    tel_base = 'WSRT1400'):
    """Build a matrix describing various linear parameters

    This function is chiefly valuable to express the effects of
    inter-instrument jumps.
    """
    year_length = 365.256363004 # sidereal year in units of SI days

    non_orbital_basis = []
    names = []
    if const:
        non_orbital_basis.append(np.ones_like(mjds))
        names.append("const")
    if P:
        non_orbital_basis.append(f0**(-1)*((mjds-pepoch)*86400))
        names.append("f0error")
    if Pdot:
        non_orbital_basis.append(f0**(-1)*0.5*((mjds-pepoch)*86400)**2)
        names.append("f1error")
    if derivs is None:
        if position:
            non_orbital_basis.extend([1e-6*np.cos(2*np.pi*mjds/year_length),
                                      1e-6*np.sin(2*np.pi*mjds/year_length)])
            names += ["pos_cos", "pos_sin"]
        if proper_motion:
            non_orbital_basis.extend(
                [(mjds-pepoch)*np.cos(2*np.pi*mjds/year_length),
                 (mjds-pepoch)*np.sin(2*np.pi*mjds/year_length)])
            names += ["pm_cos", "pm_sin"]
        if parallax:
            non_orbital_basis.extend([np.cos(4*np.pi*mjds/year_length),
                                      np.sin(4*np.pi*mjds/year_length)])
            names += ["px_cos", "px_sin"]
    else: # t2_astrometry
        new_names = []
        if position:
            new_names += ['d_RAJ','d_DECJ']
        if proper_motion:
            new_names += ['d_PMRA','d_PMDEC']
        if parallax:
            new_names += ['d_PX']
        names += new_names
        for n in new_names:
            non_orbital_basis.append(derivs[n])
    if jumps:
        debug("base telescope %s from list of %s" % (tel_base,tel_list))
        tl2 = list(tel_list)[:]
        tl2.remove(tel_base)
        tel_index = np.array([tel_list.index(t) for t in tl2])
        non_orbital_basis.append(tel_index[:,None]==tels[None,:])
        names += ["j_%s" % t for t in tl2]
    non_orbital_basis = np.vstack(non_orbital_basis).T
    return non_orbital_basis, names


def tcb_to_tdb(tcb,base_mjd=0):
    # pulled from tempo2 photons plugin
    IAU_TEPH0 = (-6.55e-5/86400)
    IAU_K = 1.550519768e-8
    IAU_KINV = 1.55051974395888e-8 # 1-1/(1-IAU_K)
    IFTE_MJD0 = 43144.0003725-base_mjd
    return tcb-IAU_K*(tcb-IFTE_MJD0)+IAU_TEPH0

def compute_orbit(parameter_dict, times, keep_states=True):
    """Compute an orbit given a dict of trial parameters

    Inputs:
    parameter_dict - a dictionary of parameters describing the
            initial conditions and physics of the problem
        acosi_i, et cetera - Keplerian parameters specifying the
            initial conditions
        ppn_mode - flag specifying how the problem physics are
            set up; determines how and which additional parameters
            are used
        use_quad - if set and True, quad precision is used
        special - include special relativistic time dilation (default True)
        general - include general relativistic time dilation (default True)
        tol - fractional and absolute tolerance parameter for ODE solver;
            defaults to 1e-16, about the smallest that works without
            quad precision
        time_reciprocal - whether to use (gamma-1) or -(1-1/gamma)

    times - the barycentric times at which to evaluate the
            orbit; zero denotes the initial time

    Outputs:
    orbit - a dictionary describing the resulting orbit
        parameter_dict - the dict of input parameters
        times - the input times
        t_bb - the barycentric times the positions are evaluated at;
            should be equal to 'times'
        t_d - the dynamical times
        t_psr - the pulsar proper times
        states - the system state (21 or 22 floating point values)
            at each time listed above
        n_evaluations - the number of RHS evaluations required
        time - the wall-clock time (in seconds) taken to integrate
            this orbit

    This function takes a set of initial conditions and evolves
    them forward using an ODE integrator. The initial conditions
    are specified as Keplerian parameters (see the module kepler);
    this function converts them to orbital state vectors in Cartesian
    coordinates in order to start the integrator. The system is
    evaluated at a list of user-provided times. These times are the
    arrival times of pulses at the (Solar) system barycenter, which
    differs from the time axis of the differential equation integrator.
    This also differs from the pulsar proper time. All three times are
    returned by this function, along with the system state at each
    time.

    In addition to specifying the initial conditions of the problem,
    the input parameter_dict also specifies the physics to be used,
    by way of the key ppn_mode. Values of this key include:
        False or None - Plain Newtonian mechanics
        "GR" - General Relativity, in a 1PN approximation
        "heavysimple" - a parameterized post-Newtonian model described by:
            dgamma - the PPN theory parameter gamma minus one
            dbeta - the PPN theory parameter beta minus one
            delta - gravitational mass of the pulsar divided by inertial
                mass of the pulsar, minus one
        "GRtidal" - General Relativity including classical tidal
            and rotational distortion of the inner companion,
            parameterized by:
            Omega - the rotation rate of the inner companion as a
                fraction of corotation
            Rc - the companion radius, in light-seconds
            k2 - the tidal Love number
    """
    start = time.time()

    debug("Running compute_orbit with parameter_dict %s" % parameter_dict)

    delta = parameter_dict.get('delta',0)
    lan = parameter_dict.get('lan',0)
    pm_x = parameter_dict.get('pm_x',0)
    pm_y = parameter_dict.get('pm_y',0)

    parameters = np.asarray([parameter_dict[p]
                for p in kepler.three_body_parameters_measurable[:14]])
    bbats = np.asarray(times)

    o = dict(parameter_dict=parameter_dict, times=times, n_evaluations=0)
    tol = parameter_dict.get('tol', 1e-16)
    use_quad = parameter_dict.get('use_quad',False)
    ppn_mode = parameter_dict.get('ppn_mode',None)
    shapiro = parameter_dict.get('shapiro',True)
    special = parameter_dict.get('special',True)
    general = parameter_dict.get('general',True)
    time_reciprocal = parameter_dict.get('time_reciprocal',False)
    matrix_mode = parameter_dict.get('matrix_mode',0)
    debug("PPN mode is %s" % ppn_mode)
    debug("Running compute_orbit from time %s to %s with tol %s"
              % (times[0],times[-1],tol))

    try:
        debug("Constructing initial conditions")
        initial_values, jac = kepler.kepler_three_body_measurable(
                *(list(parameters)+[lan,np.zeros(3),np.zeros(3),0]))
    except (ValueError, RuntimeError): # bogus system parameters
        debug("Initial conditions bogus, returning nonsense")
        if special or general:
            ls = 22
        else:
            ls = 21
        if keep_states:
            o['states'] = np.random.uniform(0, 1e40, (len(bbats),ls))
        o["initial_values"] = np.random.uniform(0, 1e40, ls)
        o['t_d'] = np.random.uniform(0, 1e40, len(bbats))
        o['t_bb'] = np.random.uniform(0, 1e40, len(bbats))
        o['t_psr'] = np.random.uniform(0, 1e40, len(bbats))
        o['time'] = 0
        return o

    n = len(bbats)
    bbats = np.copy(bbats)
    bbats[bbats<0] = 0 # avoid epoch problems during wild guessing
    if len(bbats)<n:
        debug("bbats trimmed from %d to %d" % (n,len(bbats)))
    in_order = not np.any(np.diff(bbats)<0)
    if not in_order:
        debug("TOAs have become out of order, sorting")
        ix = np.argsort(bbats)
        bbats = bbats[ix]

    debug("setting up RHS")
    if ppn_mode is None:
        rhs = quad_integrate.KeplerRHS(special=special, general=general,
                                       delta=delta,pm_x=pm_x,pm_y=pm_y,
                                       time_reciprocal=time_reciprocal)
    elif ppn_mode=='GR':
        rhs = quad_integrate.KeplerRHS(special=special, general=general,
            ppn_motion=True,matrix_mode=matrix_mode,pm_x=pm_x,pm_y=pm_y,
                                       time_reciprocal=time_reciprocal)
    elif ppn_mode=='heavypsr':
        delta = parameter_dict['delta'] # M(G) = (1+delta) M(I)
        dgamma = parameter_dict['dgamma']
        dbeta = parameter_dict['dbeta']
        #FIXME: all these three-body terms suck
        dmbeta = parameter_dict['dmbeta']
        dmpbeta = parameter_dict['dmpbeta']
        dlambda = parameter_dict['dlambda']
        mgammafac = (1+dgamma-delta)/(1+dgamma)
        rhs = quad_integrate.KeplerRHS(special=special, general=general,
            gamma=1+dgamma, beta=1+dbeta,
            Gamma01=(1+delta), Gamma02=(1+delta), Gamma12=1,
            Theta01=mgammafac, Theta02=mgammafac, Theta12=1,
            Gamma011=(1+dmpbeta), Gamma012=(1+dmpbeta), Gamma022=(1+dmpbeta),
            Gamma100=(1+dlambda), Gamma102=(1+dmbeta), Gamma122=1,
            Gamma200=(1+dlambda), Gamma201=(1+dmbeta), Gamma211=1,
            ppn_motion=True,matrix_mode=matrix_mode,pm_x=pm_x,pm_y=pm_y,
                                       time_reciprocal=time_reciprocal)
    elif ppn_mode=='heavysimple':
        delta = parameter_dict['delta'] # M(G) = (1+delta) M(I)
        dgamma = parameter_dict['dgamma']
        dbeta = parameter_dict['dbeta']
        mgammafac = (1+dgamma-delta)/(1+dgamma)
        rhs = quad_integrate.KeplerRHS(special=special, general=general,
            gamma=1+dgamma, beta=1+dbeta,
            Gamma01=(1+delta), Gamma02=(1+delta), Gamma12=1,
            Theta01=mgammafac, Theta02=mgammafac, Theta12=1,
            ppn_motion=True,matrix_mode=matrix_mode,pm_x=pm_x,pm_y=pm_y,
                                       time_reciprocal=time_reciprocal)
    elif ppn_mode=='GRtidal':
        Rc = parameter_dict['Rc']
        k2 = parameter_dict['k2']
        Omega = 2*np.pi*parameter_dict['Omega']/parameter_dict['pb_i']
        rhs = quad_integrate.KeplerRHS(special=special, general=general,
            Rc=Rc, k2=k2, Omega=Omega,
            ppn_motion=True,matrix_mode=matrix_mode,pm_x=pm_x,pm_y=pm_y,
                                       time_reciprocal=time_reciprocal)
    if special or general:
        initial_values = np.concatenate((initial_values, [0]))
    debug("Constructing ODE integrator")
    O = quad_integrate.ODEDelay(rhs,
           initial_values, 0,
           rtol=tol, atol=tol,
           shapiro=shapiro,
           use_quad=use_quad)
    l_t_bb = 0
    if keep_states:
        states = np.zeros((len(bbats),len(initial_values)),dtype=np.float128)
    ts = np.zeros((len(bbats),3),dtype=np.float128)
    report = 1
    for i,t_bb in enumerate(bbats):
        if i+1==report:
           debug("Computing TOA %d at t_bb=%g" % (i,t_bb))
        assert t_bb >= l_t_bb
        l_t_bb = t_bb
        O.integrate_to(t_bb)
        if i+1==report:
           debug("Extracting results")
           report *= 2
           #report += 1
        if keep_states:
            states[i]=O.x
        ts[i,0]=O.t_bb
        ts[i,1]=O.t_psr
        ts[i,2]=O.t_d

    debug("Done integration")
    if keep_states:
        o["states"] = states
    o["initial_values"] = initial_values
    o["t_bb"] = ts[:,0]
    o["t_psr"] = ts[:,1]
    o["t_d"] = ts[:,2]
    o["n_evaluations"] = O.n_evaluations
    o["time"] = time.time()-start

    if not in_order:
        debug("Unsorting results")
        for k in ["t_bb", "t_psr", "t_d"]:
            o[k][ix] = o[k].copy()
        if keep_states:
            o['states'][ix,:] = o['states'].copy()
    return o

def shapiro_delay(s_src, s_m, with_derivatives=False):
    """Compute the Shapiro delay

    This function computes the Shapiro delay due to light propagation
    past a massive object. It is not used internally in the orbit
    solver; that uses its own (C++) implementation of the Shapiro
    delay. Nevertheless this implementation may be useful as a
    cross-check, or for use from Python.
    """
    s_src = np.asarray(s_src)
    s_m = np.asarray(s_m)
    c = 86400. # lt-s per day
    dx = s_m[...,:3]-s_src[...,:3]
    ldx = np.sqrt(np.sum(dx**2,axis=-1))
    cst = -86400*2*kepler.G*c**(-3)
    r=cst*s_m[...,6]*np.log(dx[...,2]+ldx) # in s
    if with_derivatives:
        dz = np.zeros(s_src.shape[:-1]+(14,))
        dz[...,2] = -1
        dz[...,9] = 1
        dldx = np.zeros(s_src.shape[:-1]+(14,))
        dldx[...,:3] = -dx/ldx[...,None]
        dldx[...,7:10] = dx/ldx[...,None]
        d = (s_m[...,6]/(dx[...,2]+ldx))[...,None]*(dz+dldx)
        d[...,-1] = np.log(dx[...,2]+ldx)
        d *= cst
        return r, d
    else:
        return r

def shapiros(states, with_derivatives=False):
    """Compute the Shapiro delay for multiple states"""
    states = np.asarray(states)
    if with_derivatives:
        r1, d1 = shapiro_delay(states[...,:7],states[...,7:14],
            with_derivatives=True)
        r2, d2 = shapiro_delay(states[...,:7],states[...,14:21],
            with_derivatives=True)
        d = np.zeros_like(states)
        d[...,:14] += d1
        d[...,:7] += d2[...,:7]
        d[...,14:21] += d2[...,7:14]
        return r1+r2, d
    else:
        return (shapiro_delay(states[...,:7],states[...,7:14]) +
                shapiro_delay(states[...,:7],states[...,14:21]))

def fmt(x, u):
    """Format a value with uncertainty

    Given a value and its uncertainty, return a string showing the
    pair in 1.234(5)e6 format.
    """
    x = float(x)
    u = float(u)
    if u<0:
        raise ValueError("Uncertainty %g < 0" % u)
    elif u==0:
        return "%g" % x
    exponent_number = np.floor(np.log10(np.abs(x)))
    exponent_error = np.floor(np.log10(u))
    if u*10**(-exponent_error)>=2:
        ndigs = 1
    else:
        ndigs = 2
    if exponent_error>exponent_number:
        fstr = "0({:"+str(ndigs)+"d})e{:d}"
        return fstr.format(
            int(np.rint(u*10**(-exponent_error+(ndigs-1)))),
            int(exponent_error-ndigs+1))
    number_digits = int(exponent_number-exponent_error)+ndigs-1
    fstr = "{:."+str(number_digits)+"f}({:d})e{:d}"
    return fstr.format(
       x*10**(-exponent_number),
       int(np.rint(u*10**(-exponent_error+(ndigs-1)))),
       int(exponent_number))

def report(F, correlations=True, correlation_threshold=0.5):
        for n in F.parameters:
            print n, fmt(F.values[n],F.errors[n])
        M = F.matrix(True)
        for i,n in enumerate(F.parameters):
            print
            for j,m in enumerate(F.parameters):
                if j<=i:
                    continue
                c = M[i][j]
                if np.abs(c)>correlation_threshold:
                    print n, m, c


def load_best_parameter_database():
    with open("best-parameter-database.pickle","rb") as f:
       return pickle.load(f)
def save_best_parameter_database(bpd):
    os.rename("best-parameter-database.pickle",
              "best-parameter-database.pickle.%s~"
                  % time.strftime("%Y-%m-%d-%H:%M:%S"))
    with open("best-parameter-database.pickle","wb") as f:
        return pickle.dump(bpd,f)

multinest_prior=dict(
    asini_i=('range',(1.1,1.3)),
    pb_i=('range',(1.629,1.630)),
    eps1_i=('range',(-0.1,0.1)),
    eps2_i=('range',(-0.1,0.1)),
    tasc_i=('range',(0,1)),
    acosi_i=('range',(0,10)),
    q_i=('range',(0.01,1)),
    asini_o=('range',(70.,80.)),
    pb_o=('range',(325.,330.)),
    eps1_o=('range',(-0.1,0.1)),
    eps2_o=('range',(-0.1,0.1)),
    tasc_o=('range',(310,320)),
    acosi_o=('range',(0,200)),
    delta_lan=('range',(-np.pi,np.pi)),
    delta=('range',(-0.1,0.1)),
    dgamma=('normal',(0,2.3e-5)),
    dbeta=('normal',(0,3e-3)),
    d_RAJ=('range',(-1e-6,1e-6)),
    d_DECJ=('range',(-1e-6,1e-6)),
    j_AO1440=('range',(-1e-3,1e-3)),
    j_GBT1500=('range',(-1e-3,1e-3)),
    j_NCY1400=('range',(-1e-3,1e-3)),
    )

class Fitter(object):
    """Object representing a data set and model

    This object encapsulates a data set and a physics model, providing
    convenient support functions for parameter fitting.
    """

    def __init__(self, files=None, only_tels=None, tzrmjd_middle=False,
                 parfile="0337_bogus.par",
                 efac=1.3,
                 ppn_mode=None, matrix_mode=0,
                 use_quad=False, tol=1e-16,
                 fit_pos=False,
                 t2_astrometry=False,
                 fit_pm=False,
                 fit_px=False,
                 special=True,general=True,
                 kopeikin=False,
                 shapiro=True,
                 linear_jumps=False,
                 priors=(),
                 toa_mode=None):
        """Set up a Fitter object

        This loads a data set (see load_toas), optionally selects only
        the data corresponding to a subset of the telescopes, selects
        a zero for the pulse numbers, and establishes a background
        physics and best-guess parameter set.
        """

        av = inspect.getargvalues(inspect.currentframe())
        d = {a:av.locals[a] for a in av.args}
        del d['self']
        self.args = d
        bpd = load_best_parameter_database()
        try:
            k = frozenset(d.iteritems())
        except ValueError:
            raise ValueError("Unable to make hashable: %s" % d)
        if k in bpd:
            self.best_parameters = bpd[k]
        else:
            logger.warn("best_parameters not found on disk (%d available)"
                            % len(bpd))
            self.best_parameters = dict(f0=365.95336878765363162,
                                        dbeta=0,dgamma=0)

        self.base_mjd = 55920

        self.efac = efac

        self.ppn_mode = ppn_mode
        self.matrix_mode = matrix_mode
        self.special = special
        self.general = general
        self.use_quad = use_quad
        self.tol = tol
        self.fit_pos = fit_pos
        self.fit_pm = fit_pm
        self.fit_px = fit_px
        self.parfile = parfile
        self.shapiro = shapiro
        self.t2_astrometry = t2_astrometry
        self.kopeikin = kopeikin
        self.linear_jumps = linear_jumps

        self.files = files
        try:
            V = astropy.table.Table.read(files)
        except Exception:
            V = None
        if V is not None:
            self.base_mjd = 55920
            self.btoas = astropy.time.Time(np.asarray(V['btoas'][:,0]),
                                        np.asarray(V['btoas'][:,1]),
                                        format='jd',
                                        scale='tdb') # our F0 value is TCB
            self.mjds = ((self.btoas.jd1
                          -2400000.5
                          -self.base_mjd).astype(np.float128)
                          +self.btoas.jd2.astype(np.float128))
            self.pulses = np.array(V['pulse'], dtype=float)
            self.uncerts = 1e-6*np.array(V['errors'])
            self.derivs = {}
            self.tel_list = sorted(abbreviate_infos[i] for i in set(V['infos']))
            self.tels = np.array([self.tel_list.index(abbreviate_infos[V['infos'][i]])
                                  for i in range(len(V))])
            ix = np.argsort(self.mjds)
            self.ix = ix

            self.btoas = self.btoas[ix]
            self.mjds = self.mjds[ix]
            self.pulses = self.pulses[ix]
            self.uncerts = self.uncerts[ix]
            self.tels = self.tels[ix]

        elif files is not None:
            if self.parfile == "0337_bogus.par":
                outname = files+".out"
            else:
                outname = files+"_"+self.parfile+".out"
            if toa_mode=="pipeline":
                (self.mjds, self.pulses,
                 self.tel_list, self.tels,
                 self.uncerts, self.derivs,
                 self.ix) = load_pipeline_toas(
                     timfile=files+".tim",
                     parfile=self.parfile,
                     t2outfile=outname,
                     base_mjd=self.base_mjd)
            else:
                (self.mjds, self.pulses,
                 self.tel_list, self.tels,
                 self.uncerts, self.derivs,
                 self.ix) = load_toas(
                     timfile=files+".tim",
                     pulses=files+".pulses",
                     parfile=self.parfile,
                     t2outfile=outname,
                     base_mjd=self.base_mjd)
        else:
            (self.mjds, self.pulses,
             self.tel_list, self.tels,
             self.uncerts, self.derivs,
             self.ix) = load_toas(base_mjd=self.base_mjd)

        self.tel_base = 'WSRT1400'
        if self.tel_base not in self.tel_list:
            self.tel_base = self.tel_list[0]
        # Zap any TOAs before base_mjd
        c = self.mjds>0
        if only_tels is not None:
            c2 = np.zeros(len(c),dtype=bool)
            new_tels = np.zeros_like(self.tels)
            for i,t in enumerate(only_tels):
                c3 = self.tels==self.tel_list.index(t)
                new_tels[c3] = i
                c2 |= c3
            c &= c2
            if self.tel_base not in only_tels:
                self.tel_base = only_tels[0]
            self.tel_list = only_tels
            self.tels = new_tels
        self.mjds = self.mjds[c]
        if self.pulses is not None:
            self.pulses = self.pulses[c]
        self.tels = self.tels[c]
        self.uncerts = self.uncerts[c]
        for k in self.derivs:
            if len(self.derivs[k])==0:
                continue
            self.derivs[k] = self.derivs[k][c]

        if tzrmjd_middle=="weighted":
            mid = (np.sum(self.uncerts*self.mjds)
                   /np.sum(self.uncerts))
            i = np.searchsorted(self.mjds,mid)
            self.tzrmjd_base = self.mjds[i]
            if self.pulses is not None:
                self.pulses -= self.pulses[i]
        elif tzrmjd_middle=="auto":
            self.tzrmjd_base = None
        elif tzrmjd_middle:
            i = len(self.mjds)//2
            self.tzrmjd_base = self.mjds[i]
            if self.pulses is not None:
                self.pulses -= self.pulses[i]
        else:
            self.tzrmjd_base = 56100

        self.best_errors = {'acosi_i': 8.32004394868322e-08,
                            'acosi_o': 0.0001359635236049189,
                            'asini_i': 1.1530594682594488e-08,
                            'asini_o': 4.8118483561156396e-08,
                            'delta_lan': 3.155351744198565e-06,
                            'eps1_i': 1.2419019533035597e-08,
                            'eps1_o': 2.6472518384964833e-10,
                            'eps2_i': 1.6227860679295916e-08,
                            'eps2_o': 4.069936482827143e-10,
                            'f0': 2.7807472559990485e-12,
                            'f1': 1.403655726176749e-19,
                            'j_AO1350': 3.0271232838711795e-07,
                            'j_AO1440': 3.618571273879266e-08,
                            'j_AO327': 2.1841028352964986e-07,
                            'j_GBT1500': 3.982109983409224e-08,
                            'j_GBT350': 5.647246068277363e-07,
                            'j_GBT820': 1.5625781584242034e-07,
                            'j_WSRT350': 7.019612853617829e-07,
                            'pb_i': 6.293362566635579e-11,
                            'pb_o': 6.319266859013128e-08,
                            'q_i': 1.077186204690244e-08,
                            'tasc_i': 7.451846424579363e-09,
                            'tasc_o': 3.5744992868852454e-08,
                            'tzrmjd': 6.894185939775915e-13,
                            'delta': 1e-8,
                            'dgamma': 1e-4,
                            'dbeta': 1e-4,
                            'dmbeta': 1e-4,
                            'dmpbeta': 1e-4,
                            'dlambda': 1e-4,
                            'Omega': 1e-2,
                            'd_RAJ': 1e-8,
                            'd_DECJ': 1e-8,
                            'd_PX': 1,
                            'd_PMRA': 1e-10,
                            'd_PMDEC': 1e-10,
                            'Rc': 1e-2,
                            'k2': 1e-4,
                            'lan': 0.1,
                            'pm_x':1e-10,
                            'pm_y':1e-10,
                            'ppn_mode':ppn_mode}
        self.parameters = ['asini_i', 'pb_i', 'eps1_i', 'eps2_i', 'tasc_i',
                           'acosi_i', 'q_i',
                           'asini_o', 'pb_o', 'eps1_o', 'eps2_o', 'tasc_o',
                           'acosi_o', 'delta_lan',
                           'tzrmjd', 'f0', 'f1']
        if not self.t2_astrometry:
            if fit_pos and 'pos_cos' not in self.best_parameters:
                self.best_parameters['pos_cos'] = 0
                self.best_parameters['pos_sin'] = 0
                self.best_errors['pos_cos'] = 1
                self.best_errors['pos_sin'] = 1
        else:
            if fit_pos and 'd_RAJ' not in self.best_parameters:
                self.best_parameters['d_RAJ'] = 0
                self.best_parameters['d_DECJ'] = 0
            if fit_px and 'd_PX' not in self.best_parameters:
                self.best_parameters['d_PX'] = 0
                self.best_errors['d_PX'] = 1
            if fit_pm and 'd_PMRA' not in self.best_parameters:
                self.best_parameters['d_PMRA'] = 0
                self.best_parameters['d_PMDEC'] = 0
        if self.kopeikin:
            conv = 1.3273475184381547e-11 # mas/year to rad/day
            #FIXME: get these from par file
            self.best_parameters['pm_x'] = -3.56*conv
            self.best_parameters['pm_y'] = 3.90*conv
            if 'lan' not in self.best_parameters:
                self.best_parameters['lan'] = 0
            self.parameters.extend(['lan'])

        self.best_parameters['shapiro'] = self.shapiro

        self.best_parameters['ppn_mode'] = ppn_mode
        if ppn_mode=='heavypsr':
            self.parameters.extend([
                'delta','dgamma','dbeta','dmbeta','dmpbeta','dlambda'])
        elif ppn_mode=='heavysimple':
            self.parameters.extend(['delta','dgamma','dbeta'])
        elif ppn_mode=='GRtidal':
            self.parameters.extend(['Omega','Rc','k2'])
        self.best_parameters['use_quad'] = self.use_quad
        self.best_parameters['tol'] = self.tol
        if self.tzrmjd_base is None:
            self.best_parameters.pop('tzrmjd',None)
            self.best_parameters.pop('f1',None)
            #del self.best_parameters['f0']
            del self.parameters[self.parameters.index('tzrmjd')]
            del self.parameters[self.parameters.index('f0')]
            del self.parameters[self.parameters.index('f1')]
        if isinstance(self.efac, tuple):
            edict = dict(self.efac)
            self.efac = 1.
            self.raw_uncerts = self.uncerts.copy()
            for i,t in enumerate(self.tel_list):
                if t not in edict:
                    continue
                c = self.tels == i
                self.uncerts[c] *= edict[t]
        self.phase_uncerts = self.uncerts*self.best_parameters['f0']
        if t2_astrometry:
            derivs=self.derivs
        else:
            derivs=None
        self.jmatrix, self.jnames = trend_matrix(
            self.mjds, self.tel_list, self.tels,
            tel_base=self.tel_base,
            position=self.fit_pos,
            proper_motion=self.fit_pm,
            parallax=self.fit_px,
            derivs=self.derivs,
            const=False, P=False, Pdot=False, jumps=True)

        if not self.linear_jumps:
            self.parameters += self.jnames
        self.priors = frozenset(priors)
        self.last_p = None
        self.last_orbit = None

    def bootstrap(self):
        """Replace the observations with a "new" set chosen with replacement

        This is a utility function for error estimation using the bootstrap
        algorithm. It generates an ostensibly different set of observations
        by choosing from the available observations with replacement. The
        modified object can then be used to obtain a new set of best-fit
        parameters. The scatter in these values over many runs provides
        an error estimate.
        """
        ix = np.random.randint(0,len(self.mjds),len(self.mjds))
        ix.sort()
        self.mjds = self.mjds[ix]
        self.pulses = self.pulses[ix]
        self.uncerts = self.uncerts[ix]
        self.phase_uncerts = self.phase_uncerts[ix]
        self.tels = self.tels[ix]
        for k in self.derivs:
            if len(self.derivs[k])==0:
                continue
            self.derivs[k] = self.derivs[k][ix]
        self.jmatrix, self.jnames = trend_matrix(
            self.mjds, self.tel_list, self.tels,
            tel_base=self.tel_base,
            position=self.fit_pos,
            proper_motion=self.fit_pm,
            parallax=self.fit_px,
            derivs=self.derivs,
            const=False, P=False, Pdot=False, jumps=True)
        self.last_p = None
        self.last_orbit = None

    def compute_orbit(self, p):
        debug("Started compute_orbit for %s" % repr(p))
        if p!=self.last_p:
            debug("compute_orbit cache miss, running calculation")
            jumps = np.dot(self.jmatrix,
                               np.array([p.get(n,0) for n in self.jnames]))
            debug("Calling compute_orbit")
            o = compute_orbit(p,
                    (self.mjds)-(jumps/86400.).astype(np.float128),
                    keep_states=True)
            debug("Back from compute_orbit after time %s (%d evaluations)"
                      % (o['time'],o['n_evaluations']))
            self.last_p = p
            self.last_orbit = o
        else:
            debug("compute_orbit cache hit")
        return self.last_orbit
    def residuals(self, p=None, linear_jumps=None, marginalize=False):
        """Compute the phase residuals corresponing to a parameter dict

        Given a set of parameters, compute the orbit, then evaluate
        the predicted pulsar phase at the time each pulse was observed;
        return the difference between this and the integer pulse number
        at which the pulse was emitted.

        In addition to the parameters used by compute_orbit (q.v.) there
        are a number of additional parameters that are read from p:
            j_* - time difference between telescopes
            pos_cos, pos_sin - time delays due to position errors
            f0 - pulsar spin frequency as of self.base_mjd
            f1 - pulsar spin frequency derivative
            tzrmjd - zero of pulse phase
        If some of the last three are missing, their best-fit values
        will automatically be computed and their effect subtracted.

        If the marginalize parameter is True, this function will also
        return half the logarithm of the determinant of the matrix of
        normal equations. This quantity should be subtracted from the
        logp value to correctly perform analytical marginalization over
        these linear parameters.
        """
        debug("Started residuals for %s" % repr(p))
        if linear_jumps is None:
            linear_jumps = self.linear_jumps
        if p is None:
            p = self.best_parameters
        o = self.compute_orbit(p)
        t_psr_s = o['t_psr']*86400.
        if 'tzrmjd' in p and 'f1' in p and 'f0' in p:
            debug("Using tzrmjd f0 and f1 from parameters")
            if 'tzrmjd_base' in p:
                tzrmjd_base = p['tzrmjd_base']
            else:
                tzrmjd_base = self.tzrmjd_base
            tzrmjd_s = (p['tzrmjd']+(tzrmjd_base-self.base_mjd))*86400
            # assume PEPOCH is self.base_mjd
            phase = p['f0']*t_psr_s+p['f1']*t_psr_s**2/2.
            phase -= p['f0']*tzrmjd_s+p['f1']*tzrmjd_s**2/2.
            if marginalize:
                return phase-self.pulses, 0
            else:
                return phase-self.pulses
        else:
            debug("Setting up linear least-squares fitting")
            b = self.pulses.copy()
            At = [np.ones(len(self.pulses),dtype=t_psr_s.dtype),
                  (t_psr_s/t_psr_s[-1]), (t_psr_s/t_psr_s[-1])**2]
            if linear_jumps:
                for i in range(1,len(self.tel_list)):
                    At.append(self.tels==i)
            A = np.array(At).T
            for i in range(3):
                debug("Linear least-squares iteration %d" % i)
                x, rk, res, s = scipy.linalg.lstsq(
                    A/self.phase_uncerts[:,None],
                    b/self.phase_uncerts)
                if not np.all(np.isfinite(x)):
                    error("Warning: illegal value appeared "
                              "in least-squares fitting: %s" % x)
                    break
                b -= np.dot(A,x)
                debug("A[0]: %s b[0]: %s" % (A[0],b[0]))
                debug("x: %s" % x)
                debug("Linear least-squares residual RMS %g"
                          % np.sqrt(np.mean(b**2)))
            debug("Done linear least-squares")
            if marginalize:
                As = A/self.phase_uncerts[:,None]
                s, m = np.linalg.slogdet(np.dot(As.T,As).astype(float))
                return -b, 0.5*m
            else:
                return -b
    def compute_linear_matrix(self, p=None, linear_jumps=None, t_psr=None):
        debug("Computing linear matrix")
        if linear_jumps is None:
            linear_jumps = self.linear_jumps
        if p is None:
            p = self.best_parameters
        if t_psr is None:
            o = self.compute_orbit(p)
            t_psr = o['t_psr']
        t_psr_s = t_psr*86400.
        At = [np.ones(len(t_psr_s)),
                  t_psr_s/t_psr_s[-1],
                  0.5*(t_psr_s/t_psr_s[-1])**2]
        lp = ['tzrmjd','f0','f1']
        if linear_jumps:
            tl2 = list(self.tel_list)[:]
            tl2.remove(self.tel_base)
            tel_index = np.array([self.tel_list.index(t) for t in tl2])
            for t in tel_index:
                At.append(self.tels==t)
                lp.append("j_"+self.tel_list[t])
        A = np.array(At).T
        assert len(lp)==A.shape[1]
        return A, lp

    def compute_linear_parts(self, p=None, linear_jumps=None, t_psr=None):
        debug("Computing linear parts")
        A, lp = self.compute_linear_matrix(p,linear_jumps,t_psr)
        b = self.pulses.copy()
        r = np.zeros(A.shape[1],dtype=np.float128)
        for i in range(3):
            x, rk, res, s = scipy.linalg.lstsq(A/self.phase_uncerts[:,None],
                                               b/self.phase_uncerts)
            b -= np.dot(A,x)
            debug("A[0]: %s b[0]: %s" % (A[0],b[0]))
            debug("x: %s" % x)
            debug("residual %f" % np.sum((b/self.phase_uncerts)**2))
            debug("residual RMS %f" % np.sqrt(np.mean(b**2)))
            r += x
        if t_psr is None:
            o = self.compute_orbit(p)
            t_psr = o['t_psr']
        t_psr_s = t_psr*86400.
        f0 = r[1]/t_psr_s[-1]
        f1 = r[2]/t_psr_s[-1]**2
        def err(tzrmjd_s):
            phase = f0*t_psr_s+f1*t_psr_s**2/2.
            phase -= f0*tzrmjd_s+f1*tzrmjd_s**2/2.
            if linear_jumps:
                for i, t in enumerate(tel_index):
                    phase += r[3+i]*(self.tels==t)
            rr = phase-self.pulses
            rr /= self.phase_uncerts
            return np.sum(rr**2)

        debug("Minimizing to find tzrmjd")
        # with this ridiculous tol value it runs until no further
        # improvement is possible
        tzrmjd_s = scipy.optimize.brent(err,
            brack=(0,(self.mjds[-1])*86400),
            tol=1e-160)

        debug("done")
        d = dict(f0=f0, f1=f1, tzrmjd_base=self.base_mjd,
                    tzrmjd=tzrmjd_s/86400)
        if linear_jumps:
            for i,t in enumerate(tel_index):
                n = "j_"+self.tel_list[t]
                d[n] = p.get(n,0)-r[3+i]/f0
        return d

    def lnprob(self, p, marginalize=True):
        """Return the log-likelihood of the fit"""
        if marginalize:
            r, m = self.residuals(p,marginalize=marginalize)
        else:
            r = self.residuals(p,marginalize=marginalize)
            m = 0
        r = r/self.phase_uncerts/self.efac
        return -0.5*np.sum(r**2)-m

    def lnprior(self, p):
        """Return the log-likelihood of the prior

        Under normal operation, an MCMC algorithm simply adds the
        prior log-likelihood to the log-likelihood coming from the
        fit itself. But a parallel-tempering MCMC algorithm would
        rescale the fit's log-likelihood but not the prior, so this
        is separate.

        These priors are based on independent measurements, in most
        cases from previous GR tests.
        """
        l = 0
        if 'dbeta' in self.priors: # Mercury periastron advance
            l += (p['dbeta']/3e-3)**2
        if 'dgamma' in self.priors: # Cassini tracking
            l += (p['dgamma']/2.3e-5)**2
        if 'delta' in self.priors: # wide binary MSPs
            l += (p['delta']/6e-3)**2
        if 'pm_x' in self.priors: # 250 km/s
            l += (p['pm_x']/1e-9)**2
        if 'pm_y' in self.priors: # 250 km/s
            l += (p['pm_y']/1e-9)**2
        return -l/2.

    def chi2(self, p):
        # FIXME: should lnprior be here?
        return -2*self.efac**2*(self.lnprob(p, marginalize=False)
                                    + self.lnprior(p))

    def dof(self):
        # FIXME: linear part?
        return len(self.mjds) - len(self.parameters)

    def make_mfun(self):
        args = ", ".join(self.parameters)
        argdict = ("dict("
                       + ", ".join("%s=%s" % (p,p) for p in self.parameters)
                       + ", **moredict)")
        #lamstr = ("lambda {args}: "
        #    "np.sum((self.residuals({argdict})/self.phase_uncerts)**2)"
        #    .format(**locals())
        lamstr = ("lambda {args}: -2*self.efac**2*(self.lnprob({argdict}, "
                      "marginalize=False)+self.lnprior({argdict}))"
                      .format(**locals()))
        #print lamstr
        #print locals()
        g = globals().copy()
        g['self']=self
        g['moredict'] = dict(
            ppn_mode = self.ppn_mode,
            matrix_mode = self.matrix_mode,
            special = self.special,
            general = self.general,
            use_quad = self.use_quad,
            shapiro = self.shapiro,
            tol = self.tol,
            #FIXME: not fitting for pm_x
            pm_x = self.best_parameters.get('pm_x',0),
            pm_y = self.best_parameters.get('pm_y',0),
            )
        return eval(lamstr, g)

    def fit(self, start=None, method="simplex"):
        if start is None:
            start = np.zeros(len(self.parameters))
        last_offsets = start
        console_print("\t\t|\t" + "\t".join([p for p in self.parameters]))
        cache = {} # Powell often reuses points
        def minfunc(offsets,last_offsets=last_offsets):
            to = tuple(offsets)
            if to in cache:
                return cache[to]
            p = self.best_parameters.copy()
            for n,o in zip(self.parameters, offsets):
                p[n] += o*self.best_errors[n]
            r = -2*self.efac**2*(self.lnprob(p)+self.lnprior(p))
            console_print("%.7g\t|\t" % r + "\t".join(["%.8g" % o for o in offsets-last_offsets]))
            last_offsets[:]=offsets
            cache[to] = r
            return r
        if method=="simplex":
            # Something is wonky about the termination criteria
            xopt = scipy.optimize.fmin(minfunc,start,ftol=1e-2,xtol=1.0)
        elif method=="powell":
            xopt = scipy.optimize.fmin_powell(minfunc,start,ftol=1e-6,xtol=0.1)
        p = self.best_parameters.copy()
        for n,o in zip(self.parameters, offsets):
            p[n] += o*self.best_errors[n]
        return p

class Model(object):

    def __init__(self,
            parfile="0337_bogus.par",
            physics="GR",
            tol=1e-16,
            fit_pos=True,
            fit_pm=False,
            fit_px=False,
            ddmx_points=(),
            telescopes=('WSRT','GBT','AO'),
            n_fd=0,
            efac=1,
            equad=0,
            ecorr=0):
        """Set up a Model object"""
        # Save the creation arguments so they can be used to index
        # a database of best-fit values
        av = inspect.getargvalues(inspect.currentframe())
        d = {a:av.locals[a] for a in av.args}
        del d['self']
        try:
            d['ddmx_points'] = tuple(ddmx_points)
        except ValueError:
            # presumably not a list/array
            pass
        self.args = d

        # FIXME: get approximate f0 from par file
        self.approx_f0 = approx_f0

        self.parfile = parfile
        self.physics = physics
        self.tol = tol
        self.fit_pos = fit_pos
        self.fit_pm = fit_pm
        self.fit_px = fit_px
        self.ddmx_points = ddmx_points
        self.n_fd = n_fd
        self.efac = efac
        self.equad = equad
        self.ecorr = ecorr
        self.telescopes = telescopes
        self.orbital_parameters = [
            'asini_i', 'pb_i', 'eps1_i', 'eps2_i', 'tasc_i',
            'acosi_i', 'q_i',
            'asini_o', 'pb_o', 'eps1_o', 'eps2_o', 'tasc_o',
            'acosi_o', 'delta_lan']
        self.nonlinear_parameters = []
        if physics=="heavysimple":
            self.parameters.extend(['dgamma','dbeta','delta'])
        elif physics=="newtondelta":
            self.parameters.append(['delta'])

        self.linear_parameters = ["phase_0", "f0", "f1"]

        if self.fit_pos:
            self.linear_parameters.extend(['d_RAJ','d_DECJ'])
        if self.fit_pm:
            self.linear_parameters.extend(['d_PMRA','d_PMDEC'])
        if self.fit_px:
            self.linear_parameters.append('d_PX')
        for t in self.telescopes[1:]:
            self.linear_parameters.append("j_%s" % t)
        for (i,d) in enumerate(ddmx_points):
            self.linear_parameters.append("DMX_%d" % i)
        for i in range(n_fd):
            self.linear_parameters.append("FD_%d" % i)


class DataSet(object):

    def __init__(self, toas, parfile="0337_bogus.par"):
        pass

class Problem(object):

    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.linear_matrix = np.zeros((len(self.model.linear_parameters),
                                       len(self.dataset)))
        self.last_orbit = None
        self.last_orbit_parameters = None

    def compute_orbit(self, parameters):
        pass


def console_print(s):
    from IPython.utils.io import raw_print
    raw_print(s)