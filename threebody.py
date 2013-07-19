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

def load_toas(timfile = '0337+17.tim',
              pulses = '0337+17.pulses',
              parfile = '0337_bogus.par',
              tempo2_program = '/home/aarchiba/software/tempo2/tempo2/tempo2',
              tempo2_dir = '/home/aarchiba/software/tempo2/t2dir',
              t2outfile = '0337+17.out'):
    def mjd_fromstring(s):
        i,f = s.split(".")
        i = int(i)
        f = np.float128("0."+f)
        return i+f
    try:
        o = open(t2outfile).read()
    except IOError:
        import subprocess, os
        e = os.environ.copy()
        e['TEMPO2'] = tempo2_dir
        o = subprocess.check_output([tempo2_program,
            "-output", "general2", "-s", "OUTPUT {bat} {freq} {err}\n",
            "-tempo1",
            "-f", parfile, timfile], env=e)
        with open(t2outfile,"wt") as f:
            f.write(o)
    t2_bats = []
    freqs = []
    errs = []
    for l in o.split("\n"):
        if not l.startswith("OUTPUT"):
            continue
        t2_bats.append(mjd_fromstring(l.split()[1]))
        freqs.append(float(l.split()[2]))
        errs.append(float(l.split()[3]))
    t2_bats = np.array(t2_bats)
    errs = np.array(errs)*1e-6 # convert to s
    pulses = np.loadtxt(pulses,dtype=np.float128)
    tels = []
    tel_list = []
    def pick_tel(f):
        if 348<f<352:
            tel = 'GBT350'
        elif 326<f<328:
            tel = 'AO327'
        elif 342<f<346:
            tel = 'WSRT350'
        elif 1357.8<f<1358.5 or 1449.8<f<1450.5:
            tel = 'AO1350'
        elif 1485<f<1580:
            tel = 'GBT1500'
        elif 1439<f<1440.1:
            tel = 'AO1440'
        elif 1330<f<1390:
            tel = 'WSRT1400'
        elif 810<f<830:
            tel = 'GBT820'
        else:
            tel = 'mystery'
        return tel
    for f in freqs:
        tel = pick_tel(f)
        if tel not in tel_list:
            tel_list.append(tel)
    tel_list.sort()
    for f in freqs:
        tel = pick_tel(f)
        tels.append(tel_list.index(tel))
    tels = np.array(tels)

    ix = np.argsort(t2_bats)
    return t2_bats[ix], pulses[ix], tel_list, tels[ix], errs[ix]

best_parameters = [
	1.2175286574040021089 ,
	1.6294017424245676595 ,
	0.00068569767885231061913 ,
	-9.167915208909978898e-05 ,
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
    const=True, P=True, Pdot=True, jumps=True,
    position=False, proper_motion=False, parallax=False,
    f0 = 365.9533436144258189, pepoch = 56100, mjdbase = 55920,
    tel_base = 'WSRT1400'):
    year_length = 365.2425

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
    if jumps:
        tl2 = tel_list[:]
        tl2.remove(tel_base)
        tel_index = np.array([tel_list.index(t) for t in tl2])
        non_orbital_basis.append(tel_index[:,None]==tels[None,:])
        names += ["j_%s" % t for t in tl2]
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

def compute_orbit_bbat(parameters, bbats,
        tol=1e-16, shapiro=True, special=True, general=True, delta=0):
    # FIXME: deal with epoch not at the beginning

    start = time.time()

    parameters = np.asarray(parameters)
    bbats = np.asarray(bbats)

    o = dict(parameters=parameters, bbats=bbats,
            tol=tol, shapiro=shapiro, special=special, general=general)

    try:
        initial_values, jac = kepler.kepler_three_body_measurable(
                *(list(parameters)+[0,np.zeros(3),np.zeros(3),0]))
    except (ValueError, RuntimeError): # bogus system parameters
        if special or general:
            ls = 22
        else:
            ls = 21
        o['states'] = np.random.uniform(0, 1e40, (len(bbats),ls))
        o['t_d'] = np.random.uniform(0, 1e40, len(bbats))
        o['t_bb'] = np.random.uniform(0, 1e40, len(bbats))
        o['t_psr'] = np.random.uniform(0, 1e40, len(bbats))
        return o

    bbats = np.copy(bbats)
    bbats[bbats<0] = 0 # avoid epoch problems during wild guessing
    in_order = not np.any(np.diff(bbats<0))
    if not in_order:
        ix = np.argsort(bbats)
        bbats = bbats[ix]

    rhs = quad_integrate.KeplerRHS(special=special, general=general, delta=delta)
    if special or general:
        initial_values = np.concatenate((initial_values, [0]))
    states = []
    ts = []
    O = quad_integrate.ODEDelay(rhs,
           initial_values, 0,
           rtol = tol, atol = tol)
    l_t_bb = 0
    for t_bb in bbats:
        assert t_bb >= l_t_bb
        l_t_bb = t_bb
        O.integrate_to(t_bb)
        states.append(O.x)
        ts.append((O.t_bb,O.t_psr,O.t_d))
    states = np.array(states)
    ts = np.array(ts)

    o["states"] = states
    o["t_bb"] = ts[:,0]
    o["t_psr"] = ts[:,1]
    o["t_d"] = ts[:,2]
    o["n_evaluations"] = O.n_evaluations
    o["time"] = time.time()-start

    if not in_order:
        for k in ["t_bb", "t_psr", "t_d"]:
            o[k][ix] = o[k]
    return o

def compute_orbit(parameter_dict, times, want_states=True):
    start = time.time()

    delta = parameter_dict.get('delta',0)

    parameters = np.asarray([parameter_dict[p]
                for p in kepler.three_body_parameters_measurable[:14]])
    bbats = np.asarray(times)

    o = dict(parameter_dict=parameter_dict, times=times)
    tol = parameter_dict.get('tol', 1e-16)
    use_quad = parameter_dict.get('use_quad',False)
    ppn_mode = parameter_dict.get('ppn_mode',None)
    special = parameter_dict.get('special',True)
    general = parameter_dict.get('general',True)
    matrix_mode = parameter_dict.get('matrix_mode',0)

    try:
        initial_values, jac = kepler.kepler_three_body_measurable(
                *(list(parameters)+[0,np.zeros(3),np.zeros(3),0]))
    except (ValueError, RuntimeError): # bogus system parameters
        if special or general:
            ls = 22
        else:
            ls = 21
        o['states'] = np.random.uniform(0, 1e40, (len(bbats),ls))
        o['t_d'] = np.random.uniform(0, 1e40, len(bbats))
        o['t_bb'] = np.random.uniform(0, 1e40, len(bbats))
        o['t_psr'] = np.random.uniform(0, 1e40, len(bbats))
        return o

    bbats = np.copy(bbats)
    bbats[bbats<0] = 0 # avoid epoch problems during wild guessing
    in_order = not np.any(np.diff(bbats<0))
    if not in_order:
        ix = np.argsort(bbats)
        bbats = bbats[ix]

    if ppn_mode is None:
        rhs = quad_integrate.KeplerRHS(special=special, general=general,
                                       delta=delta)
    elif ppn_mode=='GR':
        rhs = quad_integrate.KeplerRHS(special=special, general=general,
            ppn_motion=True,matrix_mode=matrix_mode)
    if special or general:
        initial_values = np.concatenate((initial_values, [0]))
    states = []
    ts = []
    O = quad_integrate.ODEDelay(rhs,
           initial_values, 0,
           rtol = tol, atol = tol)
    l_t_bb = 0
    states = np.zeros((len(bbats),len(initial_values)),dtype=np.float128)
    ts = np.zeros((len(bbats),3),dtype=np.float128)
    for i,t_bb in enumerate(bbats):
        assert t_bb >= l_t_bb
        l_t_bb = t_bb
        O.integrate_to(t_bb)
        states[i]=O.x
        ts[i,0]=O.t_bb
        ts[i,1]=O.t_psr
        ts[i,2]=O.t_d

    o["states"] = states
    o["t_bb"] = ts[:,0]
    o["t_psr"] = ts[:,1]
    o["t_d"] = ts[:,2]
    o["n_evaluations"] = O.n_evaluations
    o["time"] = time.time()-start

    if not in_order:
        for k in ["t_bb", "t_psr", "t_d"]:
            o[k][ix] = o[k]
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


class Fitter(object):

    def __init__(self, files=None, only_tels=None, tzrmjd_middle=False, use_delta=False, ppn_mode=None):
        self.base_mjd = 55920

        self.ppn_mode = ppn_mode

        self.files = files
        if files is not None:
            (self.mjds, self.pulses,
             self.tel_list, self.tels,
             self.uncerts) = load_toas(
                 timfile=files+".tim",
                 pulses=files+".pulses",
                 t2outfile=files+".out")
        else:
            (self.mjds, self.pulses,
             self.tel_list, self.tels,
             self.uncerts) = load_toas()

        # Zap any TOAs before base_mjd
        c = self.mjds>self.base_mjd
        if only_tels is not None:
            c2 = np.zeros(len(c),dtype=bool)
            for t in only_tels:
                c2 |= self.tels==self.tel_list.index(t)
            c &= c2
        self.mjds = self.mjds[c]
        self.pulses = self.pulses[c]
        self.tels = self.tels[c]
        self.uncerts = self.uncerts[c]

        if tzrmjd_middle=="weighted":
            mid = (np.sum(self.uncerts*self.mjds)
                   /np.sum(self.uncerts))
            i = np.searchsorted(self.mjds,mid)
            self.tzrmjd_base = self.mjds[i]
            self.pulses -= self.pulses[i]
        elif tzrmjd_middle:
            i = len(self.mjds)//2
            self.tzrmjd_base = self.mjds[i]
            self.pulses -= self.pulses[i]
        else:
            self.tzrmjd_base = 56100

        if ppn_mode=='GR':
            self.best_parameters = {'acosi_i': 1.4901030967375510719,
                                    'acosi_o': 91.404576971516813065,
                                    'asini_i': 1.2175265855988132598,
                                    'asini_o': 74.672701347664255593,
                                    'delta_lan': 4.4301423835961980434e-05,
                                    'eps1_i': 0.00068720390971564635379,
                                    'eps1_o': 0.035186254399129720146,
                                    'eps2_i': -9.2091753419594045115e-05,
                                    'eps2_o': -0.0034621771473608687653,
                                    'f0': 365.95336876913063498,
                                    'f1': -2.3671254865566901149e-15,
                                    'j_AO1350': 5.3185599380177846881e-05,
                                    'j_AO1440': 4.9318510611301893706e-05,
                                    'j_AO327': 6.4728773533524038381e-05,
                                    'j_GBT1500': 6.2578922197013576212e-05,
                                    'j_GBT350': 1.8857498061175077805e-05,
                                    'j_GBT820': 6.6997790068967559975e-05,
                                    'j_WSRT350': -3.6107393277047654665e-05,
                                    'pb_i': 1.6293969144593854897,
                                    'pb_o': 327.2574963302058059,
                                    'q_i': 0.1373907308547224033,
                                    'tasc_i': 0.4075188737840608819,
                                    'tasc_o': 313.93557293857620763,
                                    'tzrmjd': 0.00073697113297647244393,
                                    'delta': 0.,
                                    'ppn_mode':ppn_mode}
        else:
            # FIXME: dig up old best_parameters from no GR
            self.best_parameters = {'acosi_i': 1.4901030967375510719,
                                    'acosi_o': 91.404576971516813065,
                                    'asini_i': 1.2175265855988132598,
                                    'asini_o': 74.672701347664255593,
                                    'delta_lan': 4.4301423835961980434e-05,
                                    'eps1_i': 0.00068720390971564635379,
                                    'eps1_o': 0.035186254399129720146,
                                    'eps2_i': -9.2091753419594045115e-05,
                                    'eps2_o': -0.0034621771473608687653,
                                    'f0': 365.95336876913063498,
                                    'f1': -2.3671254865566901149e-15,
                                    'j_AO1350': 5.3185599380177846881e-05,
                                    'j_AO1440': 4.9318510611301893706e-05,
                                    'j_AO327': 6.4728773533524038381e-05,
                                    'j_GBT1500': 6.2578922197013576212e-05,
                                    'j_GBT350': 1.8857498061175077805e-05,
                                    'j_GBT820': 6.6997790068967559975e-05,
                                    'j_WSRT350': -3.6107393277047654665e-05,
                                    'pb_i': 1.6293969144593854897,
                                    'pb_o': 327.2574963302058059,
                                    'q_i': 0.1373907308547224033,
                                    'tasc_i': 0.4075188737840608819,
                                    'tasc_o': 313.93557293857620763,
                                    'tzrmjd': 0.00073697113297647244393,
                                    'delta': 0.,
                                    'ppn_mode':ppn_mode}
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
                            'delta': 1e-3,
                            'ppn_mode':ppn_mode}
        self.parameters = ['asini_i', 'pb_i', 'eps1_i', 'eps2_i', 'tasc_i',
                           'acosi_i', 'q_i',
                           'asini_o', 'pb_o', 'eps1_o', 'eps2_o', 'tasc_o',
                           'acosi_o', 'delta_lan',
                           'tzrmjd', 'f0', 'f1']
        if use_delta:
            self.parameters.append('delta')
        self.phase_uncerts = self.uncerts*self.best_parameters['f0']
        self.jmatrix, self.jnames = trend_matrix(
            self.mjds, self.tel_list, self.tels,
            const=False, P=False, Pdot=False, jumps=True)
        self.parameters += self.jnames

    def residuals(self, p):
        jumps = np.dot(self.jmatrix,np.array([p[n] for n in self.jnames]))
        o = compute_orbit(p,
                (self.mjds-self.base_mjd)-(jumps/86400.).astype(np.float128))
        t_psr_s = o['t_psr']*86400.
        tzrmjd_s = (p['tzrmjd']+(self.tzrmjd_base-self.base_mjd))*86400
        # assume PEPOCH is self.base_mjd
        phase = p['f0']*t_psr_s+p['f1']*t_psr_s**2/2.
        phase -= p['f0']*tzrmjd_s+p['f1']*tzrmjd_s**2/2.
        return phase-self.pulses

    def mfun(self, asini_i, pb_i, eps1_i, eps2_i, tasc_i,
            acosi_i, q_i,
            asini_o, pb_o, eps1_o, eps2_o, tasc_o,
            acosi_o, delta_lan,
            tzrmjd, f0, f1,
            j_AO1350,
            j_AO1440, j_AO327, j_GBT1500,
            j_GBT350, j_GBT820,
            j_WSRT350):
        ppn_mode = self.ppn_mode
        r = self.residuals(locals())
        return np.sum((r/self.phase_uncerts)**2)


