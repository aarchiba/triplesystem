import numpy as np
from collections import namedtuple
import math
import scipy.linalg
from scipy.stats import norm
from scipy import stats

import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['savefig.dpi'] = 144
import matplotlib.mlab as mlab
from matplotlib.font_manager import FontProperties

matrix_out=namedtuple("matrix",
       ["names","A","Adict"])

#-----function-1 --- make a matrix
############################################

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


#-----function-2 --- fit A using matrix function
###################################################

def linear_least_squares_cov(par_dict, Acols, mjd, phase, unc):
    A = matrix(par_dict, Acols, mjd)
    Adict=A.Adict
    names=A.names
    x, chi2, rk, s = scipy.linalg.lstsq(A.A/unc[:,np.newaxis], phase/unc)
    res = phase-np.dot(A.A,x)
    Ascaled = A.A/unc[:,np.newaxis]

    cov = scipy.linalg.pinv(np.dot(Ascaled.T,Ascaled))
    n=len(phase)
    cov_scaled = cov*chi2/n

    class Result(object):
        pass
    r = Result()
    r.names = names
    r.x = x
    r.chi2 = chi2
    r.rk = rk
    r.s = s
    r.res = rescov = scipy.linalg.pinv(np.dot(Ascaled.T,Ascaled))
    n=len(phase)
    r.A = A
    r.Adict = Adict
    r.cov = cov
    r.cov_scaled = cov_scaled

    return r


#function-3---create_plotable_arrows (arrow array)
##################################################

def plotter(par_dict, Acols, mjd, phase, unc):
    '''Calculates arrow coefficients and values for a given data set.
       Returns: object with arrows (coordinates, directions and lengths as well as 1-sigma errors) 
par_dict - dictionary with pulsar parameters
Acols - max number of inner orbital frequencies to calculate
mjd, phase, unc - parameters of data''' 

    ang = np.linspace(0,2*np.pi,100)
    circle = np.array([np.cos(ang),np.sin(ang)])
    fit = linear_least_squares_cov(par_dict, Acols, mjd, phase, unc)
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


#-----function-4 --- gives parameters of a given arrow
######################################################

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

#-----function-5 --- makes an arrow plot from arrow array
#########################################################

font0 = FontProperties()
font0.set_size('10')
font0.set_weight('normal')
font0.set_family('serif')
font0.set_style('normal')

err_red=np.array([191, 54, 12])/255.0
err_bl=np.array([21, 101, 192])/255.0


def coeff_plot(pl, scale=None, plot_unc=True, color=err_bl, units='', lentext=True):
    '''Makes arrow plot for a given (arrow_coeff) object.
       Takes: (arrow_coeff) object. 
       Returns: value of the biggest arrow in input units and draws plot
pl - object with arrow coordinates, directions and lenghts (input your units)'''
    if scale is None:
        ar_scale=np.amax(np.hypot(pl.V,pl.U))
    else:
        ar_scale=scale
    if plot_unc:
        for k in range(0,len(pl.err_X)):
            i,j = pl.err_ix[k]
            plt.plot(pl.err_X[k]/ar_scale+j,pl.err_Y[k]/ar_scale+i, color=color, lw=1, alpha=0.3)
    Q = plt.quiver(pl.X, pl.Y, pl.U, pl.V, units='x', scale=ar_scale, color=color)
    Acols=np.amax(pl.X)+1
    plt.xlim(-Acols+0.05,Acols-0.05)
    plt.ylim(-0.95,Acols-0.05)
    plt.xlabel('Inner orbit frequencies')
    plt.ylabel('Outer orbit frequencies')
    plt.gca().set_aspect('equal')
    err_red=np.array([191, 54, 12])/255.0
    if lentext:
        plt.figtext(0.15,0.2, r'the longest arrow = %f %s'%(ar_scale,units), fontproperties=font0, color=err_red)
    return ar_scale

#-----function-5 --- makes an arrow plot from res and mjd
#########################################################

def draw_plot(par_dict, Acols, mjd, phase, unc, scale=None, plot_unc=True, color=err_bl, units='', lentext=True):
    '''Makes arrow plot for a given data set.
       Returns: value of the biggest arrow and plot
par_dict - dictionary with pulsar parameters
Acols - max number of inner orbital frequencies to calculate
mjd, phase, unc - parameters of data (in units you want to get your arrow lenghts in)'''
    pl=plotter(par_dict, Acols, mjd, phase, unc)
    return coeff_plot(pl,scale=scale, plot_unc=plot_unc, color=color, units=units, lentext=lentext)



