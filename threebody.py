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


def load_data(filename="0337_delays_uncertainty.txt",
        doppler_correct=True):
    delay_list = []
    with open(filename) as f:
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
best_parameters_nogr = [
	1.217529967691550754 ,
	1.629401735908281567 ,
	-9.1680933187314077105e-05 ,
	0.00068607294788486786044 ,
	0.4074892430788351937 ,
	1.4861853088300347585 ,
	0.13744418603049468959 ,
	74.672663767844384898 ,
	327.25756096585252719 ,
	-0.0034618550185082725042 ,
	0.035186307687828952724 ,
	313.93597677179654973 ,
	91.202189778972841357 ,
	-4.4596707010768933897e-05 ,
]
best_errors_nogr = [
	1.8624739061394744e-08 ,
	3.625137710539212e-10 ,
	1.9190728493949218e-08 ,
	1.673908407209016e-08 ,
	9.816045154137255e-09 ,
	3.5860679776065874e-06 ,
	1.7327413665307784e-06 ,
	1.2327858353102708e-07 ,
	4.1809524403587833e-07 ,
	9.172363390894159e-10 ,
	6.199662972285706e-10 ,
	3.871106137895912e-07 ,
	0.0017795929143580116 ,
	7.70027304890772e-06 ,
]
def trend_matrix(mjds, tel_list, tels, 
    P=True, Pdot=True, jumps=True,
    position=False, proper_motion=False, parallax=False,
    f0 = 365.9533436144258189, pepoch = 56100, mjdbase = 55920):
    year_length = 365.2425
    
    non_orbital_basis = [np.ones_like(mjds)]
    names = ["const"]
    if P:
        non_orbital_basis.append(f0**(-1)*((mjds-pepoch)*86400))
        names.append("f0error")
    if Pdot:
        non_orbital_basis.append(f0**(-1)*0.5*((mjds-pepoch)*86400)**2)
        names.append("f1error")
    if jumps:
        non_orbital_basis.append(np.arange(1,len(tel_list))[:,None]==tels[None,:])
        names += ["j_%s" % t for t in tel_list[1:]]
    if position:
        non_orbital_basis.extend([np.cos(2*np.pi*mjds/year_length),
                                  np.sin(2*np.pi*mjds/year_length)])
        names += ["pos_cos", "pos_sin"]
    if proper_motion:
        non_orbital_basis.extend([(mjds-pepoch)*np.cos(2*np.pi*mjds/year_length),
                                  (mjds-pepoch)*np.sin(2*np.pi*mjds/year_length)])
        names += ["pm_cos", "pm_sin"]
    if parallax:
        non_orbital_basis.extend([np.cos(4*np.pi*mjds/year_length),
                                  np.sin(4*np.pi*mjds/year_length)])
        names += ["px_cos", "px_sin"]
    non_orbital_basis = np.vstack(non_orbital_basis).T
    return non_orbital_basis, names
    
def remove_trend(vec, mjds, tel_list, tels, uncerts=None,
    P=True, Pdot=True, jumps=True,
    position=False, proper_motion=False, parallax=False):
    year_length = 365.2425
    
    non_orbital_basis, names = trend_matrix(mjds, tel_list, tels, 
            P, Pdot, jumps, position, proper_motion, parallax)
    if uncerts is None:
        x, res, rk, s = scipy.linalg.lstsq(non_orbital_basis, vec)
    else:
        x, res, rk, s = scipy.linalg.lstsq(
            non_orbital_basis/uncerts[:,None], 
            vec/uncerts)
    return vec-np.dot(non_orbital_basis,x)

def compute_orbit(parameters, times, with_derivatives=False, epoch=0, 
        tol=1e-16, delta=1e-12, symmetric=False,
        shapiro=True, special=True, general=True, use_quad=False):
    # FIXME: deal with epoch not at the beginning

    start = time.time()

    parameters = np.asarray(parameters)
    times = np.asarray(times)

    o = dict(parameters=parameters, times=times, 
            tol=tol, delta=delta, symmetric=symmetric,
            shapiro=shapiro, special=special, general=general,
            use_quad=use_quad)

    try:
        initial_values, jac = kepler.kepler_three_body_measurable(
                *(list(parameters)+[0,np.zeros(3),np.zeros(3),0]))
    except (ValueError, RuntimeError): # bogus system parameters
        if special or general:
            ls = 22
        else:
            ls = 21
        o['states'] = np.random.uniform(0, 1e40, (len(times),ls))
        o['delays'] = np.random.uniform(0, 1e40, len(times))
        if shapiro:
            o['shapiro'] = np.random.uniform(0, 1e40, len(times))
        if with_derivatives:
            o['derivatives'] = np.random.normal(0, 1e40, (len(times),14))
        o['name'] = None
        return o
    vectors = jac[:,:14].T
    
    rhs = quad_integrate.KeplerRHS(special=special, general=general)
    if special or general:
        initial_values = np.concatenate((initial_values, [0]))
        vectors = np.concatenate([vectors, np.zeros(14)[:,None]], axis=1)
    states = []
    shapiro_delays = []
    if with_derivatives:
        derivatives = []
        derivative_errors = []
        O = quad_integrate.ODE(rhs,
               initial_values, 0,
               rtol = tol, atol = tol,
               vectors = vectors,
               delta = delta,
               symmetric = symmetric,
               use_quad = use_quad)
        for t in times:
            O.integrate_to(t)
            states.append(O.x)
            dx = O.dx
            derivatives.append(dx[2,:].copy())
            if special or general:
                derivatives[-1] += 86400*dx[21,:]
            if symmetric:
                derivative_errors.append(O.dx_error[2,:].copy())
                if special or general:
                    derivative_errors[-1] += 86400*O.dx_error[21,:]
            if shapiro:
                s, d = shapiros(O.x, with_derivatives=True)
                derivatives[-1] += np.dot(d, dx)
                shapiro_delays.append(s)
        derivatives = np.array(derivatives)
        derivative_errors = np.array(derivatives)
    else:
        O = quad_integrate.ODE(rhs,
               initial_values, 0,
               rtol = tol, atol = tol,
               vectors = [],
               use_quad = use_quad)
        for t in times:
            O.integrate_to(t)
            states.append(O.x)
            if shapiro:
                s = shapiros(O.x)
                shapiro_delays.append(s)
    states = np.array(states)

    o["states"] = states
    o["delays"] = states[:,2].copy()
    if with_derivatives:
        o["derivatives"] = derivatives
        if symmetric:
            o["derivative_errors"] = derivative_errors
    if special or general:
        o["einstein_delays"] = states[:,21]
        o["delays"] += 86400*states[:,21]
        # einstein already included in derivatives
    if shapiro:
        o["shapiro_delays"] = np.array(shapiro_delays)
        o["delays"] += o["shapiro_delays"]

    o["n_evaluations"] = O.n_evaluations
    o["time"] = time.time()-start

    return o

def shapiro_delay(s_src, s_m, with_derivatives=False):
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

