import numpy as np
from collections import namedtuple
import math

matrix_out=namedtuple("matrix",
       ["names","A","Adict"])


def matrix(par_dict, ll, mjd):
    Adict = {}
    pb_i = par_dict['pb_i']
    pb_o = par_dict['pb_o']
    phi_i=(2.0*math.pi)/pb_i
    phi_o=(2.0*math.pi)/pb_o
    tasc_i = par_dict["tasc_i"]
    tasc_o = par_dict["tasc_o"]
    for i in range(ll):
        for j in range(-(ll-1),ll):
            if i==0 and j<0:
                continue
            Adict[i,j,'cos'] = np.cos(i*phi_o*(mjd-tasc_o)+j*phi_i*(mjd-tasc_i))
            if (i,j)!=(0,0):
                Adict[i,j,'sin'] = np.sin(i*phi_o*(mjd-tasc_o)+j*phi_i*(mjd-tasc_i))
    names = sorted(Adict.keys())
    A = np.array([Adict[n] for n in names]).T
    return matrix_out(names=names, A=A, Adict=Adict)
