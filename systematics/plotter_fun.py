import numpy as np
import scipy.linalg
from collections import namedtuple
import math
from scipy.stats import norm
from scipy import stats
import lsqr_fit_fun as lsqr_fit

#function-3---create_plotable_arrows
####################################
plotter_out = namedtuple("plotter_out",
    ["X","Y","U","V","M","err_X","err_Y","err_ix"])

def plotter(par_dict, Acols, mjd, phase, unc):
    ang = np.linspace(0,2*np.pi,100)
    circle = np.array([np.cos(ang),np.sin(ang)])
    fit = lsqr_fit.linear_least_squares_cov(par_dict, Acols, mjd, phase, unc)
    result = dict(zip(fit.names,fit.x))
    X = []
    Y = []
    U = []
    V = []
    err_X = []
    err_Y = []
    err_ix = []
    for (i,j,f) in result.keys():
        Y.append(i)
        X.append(j)
        U.append(result[i,j,'cos'])
        err_ix.append((i,j))
        if (i,j)==(0,0):
            V.append(0)
            err_X.append(0)
            err_Y.append(0)
        else:
            V.append(result[i,j,'sin'])

            st=fit.names.index((i, j, 'cos'))
            fin=fit.names.index((i, j, 'sin'))+1
            err_M=fit.cov[st:fin,st:fin]
            assert err_M.shape==(2,2)
            L = scipy.linalg.cholesky(err_M)
            Lcircle = np.dot(L,circle)
            # print Lcircle[0][7]
            err_X.append(Lcircle[0])
            err_Y.append(Lcircle[1])

    M = np.hypot(V, U)
    return plotter_out(X=X, Y=Y, U=U, V=V, M=M, err_X=err_X, err_Y=err_Y, err_ix=err_ix)   
