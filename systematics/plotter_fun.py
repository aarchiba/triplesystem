import numpy as np
import scipy.linalg
from collections import namedtuple
import math
from scipy.stats import norm
from scipy import stats
import lsqr_fit_fun as lsqr_fit

#function-3---create_plotable_arrows
####################################

def plotter(par_dict, Acols, mjd, phase, unc):
    '''Calculates arrow coefficients and values for a given data set.
       Returns: object with arrows (coordinates, directions and lengths as well as 1-sigma errors) 
par_dict - dictionary with pulsar parameters
Acols - max number of inner orbital frequencies to calculate
mjd, phase, unc - parameters of data''' 

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
    class Result(object):
         pass
    pl = Result()
    pl.X = np.array(X)
    pl.Y = np.array(Y)
    pl.U = np.array(U)
    pl.V = np.array(V)
    pl.M = np.array(M)
    pl.err_X = np.array(err_X)
    pl.err_Y = np.array(err_Y)
    pl.err_ix = np.array(err_ix)
    return pl


def arrow_extract(pl,k,j):
    '''Takes (arrow_coeff) object and frequecncies of the arrow you want to get info about.
       Returns len=3 array: [projection_X, projection_Y, lenght].
pl - object with arrow coordinates, directions and lenghts
k - inner orbit frequency
j - outer orbit frequency''' 
    for i in range(0,len(pl.X)):
            if pl.X[i]==k and pl.Y[i]==j:
                ll=np.array([pl.U[i], pl.V[i], pl.M[i]])
    return ll
