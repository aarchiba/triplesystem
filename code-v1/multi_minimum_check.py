# -*- coding: utf-8 -*-
import os

import numpy as np
import numpy.random
import scipy.optimize

import kepler
import quad_integrate
import threebody

mode='lm'
#mode='bfgs'
delay_list = []
with open("0337_delays_2.txt") as f:
    for l in f.readlines():
        if l.startswith("#"):
            continue
        mjd, delay, tel = l.split()
        delay_list.append((float(mjd),float(delay),tel))
mjds = np.array([m for (m,d,t) in delay_list])
delays = np.array([d for (m,d,t) in delay_list])

tel_list = list(sorted(set([t for (m,d,t) in delay_list])))
tels = np.array([tel_list.index(t) for (m,d,t) in delay_list])

mjds -= delays/86400. # reomve Doppler effect

ix = np.argsort(mjds)
mjds = mjds[ix]
delays = delays[ix]

tels = tels[ix]

params = [
         1.2174747243188027,
         1.6294050499170794,
         -8.924516861524968e-05,
         0.000685253016835822,
         0.4075020223794183,
         0.7448736328511888,
         0.11700148191244052,
         74.67249772629293,
         327.2543331024813,
         -0.0034677237764728424,
         0.03518458652298544,
         313.93278525531576,
         39.4975669344576,
         0.022249717631385148
        ]

def random_parameters():
    v = {}
    for (n,p) in zip(kepler.three_body_parameters_measurable[:14],params):
        v[n] = p
    duration = mjds[-1]-mjds[0]
    v['pb_i'] += np.random.uniform(-0.001,0.001)/(duration/v['pb_i'])
    v['pb_o'] += np.random.uniform(-0.001,0.001)/(duration/v['pb_o'])
    v['asini_i'] += np.random.randn()*0.001*v['asini_i']
    v['asini_o'] += np.random.randn()*0.001*v['asini_o']
    v['acosi_i'] = v['asini_i']/np.tan(np.pi*np.random.uniform(-0.5,0.5))
    v['acosi_o'] = v['asini_o']/np.tan(np.pi*np.random.uniform(-0.5,0.5))
    v['eps1_i'] += np.random.randn()*1e-5
    v['eps2_i'] += np.random.randn()*1e-5
    v['eps1_o'] += np.random.randn()*1e-5
    v['eps2_o'] += np.random.randn()*1e-5
    v['q_i'] = (7.32+0.8*np.random.randn())**(-1)
    v['delta_lan'] = np.random.uniform(0,2*np.pi)
    v['tasc_i'] += np.random.randn()*v['pb_i']*0.01
    v['tasc_o'] += np.random.randn()*v['pb_o']*0.01
    #v['tasc_i'] = np.random.uniform(0,v['pb_i'])
    #v['tasc_o'] = np.random.uniform(0,v['pb_o'])
    for k in sorted(v.keys()):
        print k, v[k]
    return [v[n] for n in kepler.three_body_parameters_measurable[:-4]]

def fun(p):
    print "fun:",p,
    o = threebody.compute_orbit(p, mjds)
    rv = threebody.remove_trend(o['states'][:,2]-delays, mjds, tel_list, tels)
    r = np.sqrt(np.mean(rv**2))
    print r
    if mode=='lm':
        return np.asarray(rv, float)
    else:
        return r

while True:
    if not os.path.exists("minima.txt"):
        p = np.array(params)
        fp = p
    else:
        p = random_parameters()
        print "start:", p
        if mode=='lm':
            r = scipy.optimize.leastsq(fun, p)
            fp = r[0]
        elif mode=='bfgs':
            r = scipy.optimize.fmin_bfgs(fun, p)
            fp = r
        else:
            raise ValueError
    if mode=='lm':
        err = np.sqrt(np.mean(fun(fp)**2))
    else:
        err = fun(fp)
    print "finish:", fp, err
    with open("minima.txt","a") as f:
        f.write("\t".join(repr(v) for v in [err]+list(fp)+list(p)))
        f.write("\n")


