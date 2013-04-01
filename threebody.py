import os
import string
import random
from glob import glob
import time

import numpy as np
import numpy.random
import scipy.linalg

import kepler
import quad_integrate


def load_data(filename="0337_delays_2.txt",
        doppler_correct=True):
    delay_list = []
    with open("0337_delays_uncertainty.txt") as f:
        for l in f.readlines():
            if l.startswith("#"):
                continue
            mjd, delay, tel, uncert = l.split()
            delay_list.append((float(mjd),float(delay),tel,float(uncert)))
    mjds = np.array([m for (m,d,t,u) in delay_list])
    delays = np.array([d for (m,d,t,u) in delay_list])
    tel_list = list(sorted(set([t for (m,d,t,u) in delay_list])))
    tels = np.array([tel_list.index(t) for (m,d,t,u) in delay_list])
    uncerts = 1e-6*np.array([u for (m,d,t,u) in delay_list])
    if doppler_correct:
        mjds -= delays/86400.
    ix = np.argsort(mjds)
    mjds = mjds[ix]
    delays = delays[ix]
    tels = tels[ix]
    uncerts = uncerts[ix]

    return mjds, delays, tel_list, tels, uncerts


best_parameters = [
	1.2175286574040021089 ,
	1.6294017424245676595 ,
	-9.167915208909978898e-05 ,
	0.00068569767885231061913 ,
	0.40748888446707576171 ,
	1.4928413805962691032 ,
	0.13739679461335577318 ,
	74.672709181427526108 ,
	327.25754395677783318 ,
	-0.0034621278217266577593 ,
	0.035186272495204991849 ,
	313.93558318886735767 ,
	91.611372740471614128 ,
	-4.1110581185487967683e-05 ,
]
best_errors = [
	1.7556575632954052e-08 ,
	2.2817007892786326e-10 ,
	1.6568485883029365e-08 ,
	1.4557847466119638e-08 ,
	8.663674326627852e-09 ,
	2.6254576384333125e-05 ,
	1.0288730471728326e-06 ,
	2.140792343411674e-07 ,
	3.322543685198226e-07 ,
	7.601435134628213e-10 ,
	6.106381168432414e-10 ,
	2.1022233493158794e-07 ,
	0.001609442571711061 ,
	7.981798704760887e-06 ,
]


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
def remove_trend(vec, mjds, tel_list, tels, uncerts=None,
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
    if uncerts is None:
        x, res, rk, s = scipy.linalg.lstsq(non_orbital_basis, vec)
    else:
        x, res, rk, s = scipy.linalg.lstsq(
            non_orbital_basis/uncerts[:,None], 
            vec/uncerts)
    return vec-np.dot(non_orbital_basis,x)

def process_remove_trend(o):
    o["states"][:,2] = remove_trend(o["states"][:,2])

def compute_orbit(parameters, times, with_derivatives = False, epoch = 0,
        save_orbits=False, load_orbits=False, tol=1e-16, 
        shapiro=False, special=False, general=False, use_quad=False):
    if (load_orbits or save_orbits) and (shapiro or special or general1 or general2):
        raise NotImplementedError
    if shapiro and with_derivatives:
        raise NotImplementedError
    start = time.time()
    parameters = np.asarray(parameters)
    times = np.asarray(times)
    if load_orbits:
        for o in saved_orbits(times, with_derivatives):
            if np.all(o["parameters"]==parameters):
                return o

    o = dict(parameters=parameters, times=times)

    try:
        initial_values, jac = kepler.kepler_three_body_measurable(
                *(list(parameters)+[0,np.zeros(3),np.zeros(3),0]))
    except (ValueError, RuntimeError): # bogus system parameters
        if special or general:
            ls = 22
        else:
            ls = 21
        states = np.random.uniform(0, 1e40, (len(times),ls))
        o['states'] = states
        if with_derivatives:
            derivatives = np.random.normal(0, 1e40, (len(times),ls,14))
            o['derivatives'] = derivatives
        o['name'] = None
        return o
    vectors = jac[:,:14].T
    
    # FIXME: deal with epoch not at the beginning

    rhs = quad_integrate.KeplerRHS(
            special=special, general=general)
    if special or general:
        initial_values = np.concatenate((initial_values, [0]))
    states = []
    if with_derivatives:
        derivatives = []
        O = quad_integrate.ODE(rhs,
               initial_values, 0,
               rtol = tol, atol = tol,
               vectors = vectors,
               delta = 1e-10,
               use_quad = use_quad)
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
               delta = 1e-10,
               use_quad = use_quad)
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
    o["n_evaluations"] = O.n_evaluations
    o["time"] = time.time()-start

    if save_orbits:
        o["name"] = save_orbit(**o)

    return o

def shapiro_delay(s_src, s_m):
    c = 86400. # lt-s per day
    dx = s_m[...,:3]-s_src[...,:3]
    ldx = np.sqrt(np.sum(dx**2,axis=-1))
    #return -86400*2*kepler.G*s_m[...,6]*c**(-3)*np.log(1-dx[...,2]/ldx) # in s
    return -86400*2*kepler.G*s_m[...,6]*c**(-3)*np.log(dx[...,2]+ldx) # in s

def shapiros(states):
    return (shapiro_delay(states[:,:7],states[:,7:14]) +
                shapiro_delay(states[:,:7],states[:,14:21]))

def fmt(x, u):
    
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

