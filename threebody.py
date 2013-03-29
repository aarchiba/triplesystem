import os
import string
import random
from glob import glob

import numpy as np
import numpy.random
import scipy.linalg

import kepler
import quad_integrate


orbit_dir = "orbits"

def save_orbit(parameters, times, states, derivatives = None):
    try:
        os.mkdir(orbit_dir)
    except OSError:
        pass

    name = time.strftime("%Y-%m-%d-%H:%M:%S")
    name += "-"+"".join([random.choice(string.lowercase) for i in range(20)])
    name = os.path.join(orbit_dir,name)

    np.savez(name, 
            parameters = parameters,
            times = times,
            states = states,
            derivatives = derivatives)
    return name

def saved_orbits(times = None, with_derivatives = False):
    for n in sorted(glob(os.path.join(orbit_dir, "*-*-*-*:*:*-*.npz")))[::-1]:
        r = {"name":n}
        with np.load(n) as f:
            if with_derivatives and "derivatives" not in f:
                continue
            if times is not None and not np.all(times==f["times"]):
                continue
            for a in f.files:
                r[a] = f[a]
        yield r

def best_orbit(quality_evaluator, post_process = None, 
        times = None,
        with_derivatives = False):
    r = np.inf
    best = None
    for o in saved_orbits(times, with_derivatives):
        if post_process is not None:
            post_process(o)
        nr = quality_evaluator(o)
        if nr<r:
            r = nr
            best = o
    return best

def least_squared(delays):
    def _least_squared(o):
        return np.mean((o.states[:,2]-delays)**2)
    return _least_squared
def remove_trend(vec, mjds, tel_list, tels,
    P=True, Pdot=True, jumps=True,
    position=False, proper_motion=False, parallax=False):
    year_length = 365.2425
    
    non_orbital_basis = [np.ones_like(mjds)]
    if P:
        non_orbital_basis.append(mjds)
    if Pdot:
        non_orbital_basis.append(mjds**2/2.)
    if jumps:
        non_orbital_basis.append(np.arange(1,len(tel_list))[:,None]==tels[None,:])
    if position:
        non_orbital_basis.extend([np.cos(2*np.pi*mjds/year_length),
                                  np.sin(2*np.pi*mjds/year_length)])
    if proper_motion:
        non_orbital_basis.extend([mjds*np.cos(2*np.pi*mjds/year_length),
                                  mjds*np.sin(2*np.pi*mjds/year_length)])
    if parallax:
        non_orbital_basis.extend([np.cos(4*np.pi*mjds/year_length),
                                  np.sin(4*np.pi*mjds/year_length)])
    non_orbital_basis = np.vstack(non_orbital_basis).T
    x, res, rk, s = scipy.linalg.lstsq(non_orbital_basis, vec)
    return vec-np.dot(non_orbital_basis,x)
def process_remove_trend(o):
    o["states"][:,2] = remove_trend(o["states"][:,2])

def compute_orbit(parameters, times, with_derivatives = False, epoch = 0,
        save_orbits=False, load_orbits=False, tol=1e-16, 
        shapiro=False, special=False, general=False):
    if (load_orbits or save_orbits) and (shapiro or special or general1 or general2):
        raise NotImplementedError
    if with_derivatives and (special or general):
        raise NotImplementedError
    parameters = np.asarray(parameters)
    times = np.asarray(times)
    if load_orbits:
        for o in saved_orbits(times, with_derivatives):
            if np.all(o["parameters"]==parameters):
                return o

    o = dict(parameters=parameters, times=times)

    # FIXME: deal with bogus values when state is 22 long
    try:
        initial_values, jac = kepler.kepler_three_body_measurable(
                *(list(parameters)+[0,np.zeros(3),np.zeros(3),0]))
    except (ValueError, RuntimeError): # bogus system parameters
        states = np.random.uniform(0, 1e40, (len(times),21))
        #states = np.empty((len(times),21))
        #states[...] = np.inf # make least-squared come out infinite
        o['states'] = states
        derivatives = np.random.normal(0, 1e40, (len(times),21,14))
        #derivatives = np.empty((len(times),21,14))
        #derivatives[...] = np.inf
        o['derivatives'] = derivatives
        o['name'] = None
        return o
    vectors = jac[:,:14].T
    
    # FIXME: deal with epoch not at the beginning

    if special or general:
        rhs = quad_integrate.KeplerRHSRelativity(
                special=special, general=general)
        initial_values = np.concatenate((initial_values, [0]))
    else:
        rhs = quad_integrate.KeplerRHS()
    states = []
    if with_derivatives:
        derivatives = []
        O = quad_integrate.ODE(rhs,
               initial_values, 0,
               rtol = tol, atol = tol,
               vectors = vectors,
               delta = 1e-10)
        for t in times:
            O.integrate_to(t)
            states.append(O.x)
            derivatives.append(O.dx)
        derivatives = np.array(derivatives)
    else:
        O = quad_integrate.ODE(rhs,
               initial_values, 0,
               rtol = tol, atol = tol,
               vectors = [],
               delta = 1e-10)
        for t in times:
            O.integrate_to(t)
            states.append(O.x)
        derivatives = None
    states = np.array(states)

    o["states"] = states
    o["derivatives"] = derivatives
    if special or general:
        o["einstein"] = states[:,21]
    if shapiro:
        o["shapiro"] = shapiros(states)

    if save_orbits:
        o["name"] = save_orbit(**o)

    return o

def shapiro_delay(s_src, s_m):
    c = 86400. # lt-s per day
    dx = s_m[...,:3]-s_src[...,:3]
    ldx = np.sqrt(np.sum(dx**2,axis=-1))
    return -86400*2*kepler.G*s_m[...,6]*c**(-3)*np.log(1-dx[...,2]/ldx) # in s

def shapiros(states):
    return (shapiro_delay(states[:,:7],states[:,7:14]) +
                shapiro_delay(states[:,:7],states[:,14:21]))
