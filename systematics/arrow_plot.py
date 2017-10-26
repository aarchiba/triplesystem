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

def ar_coeff(par_dict, Acols, mjd, phase, unc):
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
    err_U = []
    err_V = []
    err_ix = []
    for (i,j,f) in result.keys():
        Y.append(i)
        X.append(j)
        U.append(result[i,j,'cos'])
        err_ix.append((i,j))
        if (i,j)==(0,0):
            V.append(0)
            err_U.append(0)
            err_V.append(0)
            err_X.append(0)
            err_Y.append(0)
        else:
            V.append(result[i,j,'sin'])

            st=fit.names.index((i, j, 'cos'))
            fin=fit.names.index((i, j, 'sin'))+1
            err_MM=fit.cov[st:fin,st:fin]
            err_U.append(np.sqrt(np.diag(err_MM))[0])
            err_V.append(np.sqrt(np.diag(err_MM))[1])
            assert err_MM.shape==(2,2)
            L = scipy.linalg.cholesky(err_MM)
            Lcircle = np.dot(L,circle)
            # print Lcircle[0][7]
            err_X.append(Lcircle[0])
            err_Y.append(Lcircle[1])

    M = np.hypot(V, U)
    err_M = np.hypot(err_U,err_V)
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
    pl.err_U = np.array(err_U)
    pl.err_V = np.array(err_V)
    pl.err_M = np.array(err_M)
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
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
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
    pl=ar_coeff(par_dict, Acols, mjd, phase, unc)
    return coeff_plot(pl,scale=scale, plot_unc=plot_unc, color=color, units=units, lentext=lentext)

def lstsq_with_errors(A,b,uncerts=None):
    """Solve a linear least-squares problem and return uncertainties

    This function extends `scipy.linalg.lstsq` in several ways: first,
    it supports uncertainties on each row of the least-squares problem.
    Second, `scipy.linalg.lstsq` fails if the scales of the fit
    variables are very different. This function rescales them 
    internally to improve the condition number. Finally, this function
    returns an object containing information about the uncertainties
    on the fit values; the `uncerts` attribute gives individual
    uncertainties, the `corr` attribute is the matrix of correlations,
    and the `cov` matrix is the full covariance matrix.
    """
    if len(A.shape)!=2:
        raise ValueError
    if uncerts is None:
        Au = A
        bu = b
    else:
        Au = A/uncerts[:,None]
        bu = b/uncerts
    Ascales = np.sqrt(np.sum(Au**2,axis=0))
    #Ascales = np.ones(A.shape[1])
    if np.any(Ascales==0):
        raise ValueError("zero column (%s) in A" % np.where(Ascales==0))
    As = Au/Ascales[None,:]
    db = bu
    xs = None
    best_chi2 = np.inf
    best_x = None
    for i in range(5): # Slightly improve quality of fit
        dxs, res, rk, s = scipy.linalg.lstsq(As, db)
        if rk != A.shape[1]:
            raise ValueError("Condition number still too bad; "
                             "singular values are %s"
                             % s)
        if xs is None:
            xs = dxs
        else:
            xs += dxs
        db = bu - np.dot(As, xs)
        chi2 = np.sum(db**2)
        if chi2<best_chi2:
            best_chi2 = chi2
            best_x = xs
        #debug("Residual chi-squared: %s", np.sum(db**2))
    x = best_x/Ascales # FIXME: test for multiple b
    
    class Result:
        pass
    r = Result()
    r.x = x
    r.chi2 = best_chi2
    bias_corr = A.shape[0]/float(A.shape[0]-A.shape[1])
    r.reduced_chi2 = bias_corr*r.chi2/A.shape[0]
    Atas = np.dot(As.T, As)
    covs = scipy.linalg.pinv(Atas)
    r.cov = covs/Ascales[:,None]/Ascales[None,:]
    r.uncerts = np.sqrt(np.diag(r.cov))
    r.corr = r.cov/r.uncerts[:,None]/r.uncerts[None,:]
    return r

def der_fit(data,phase,unc):
    d=np.array(data['derivatives'])[np.newaxis][0]
    cols= sorted(d.keys())
    A = np.array([d[c] for c in cols]).T
    b=phase
    r = lstsq_with_errors(A, b, unc)
    better_phase=b-np.dot(A,r.x)
    return better_phase


def der_of_par(data, par, unc):
    d=np.array(data['derivatives'])[np.newaxis][0]
    cols= sorted(d.keys())
    print np.shape(cols)

    del cols[cols.index(par)]
    A = np.array([d[c] for c in cols]).T

    print A.shape
    b=d[par]

    r = lstsq_with_errors(A, b, unc)
    der_par=(b-np.dot(A,r.x))
    return der_par

def plot_hex(par_dict, mjd,res, size, colorbar=True):
    my_width=6
    mar_size=my_width*0.33
    lab_size=my_width*1.7
    tick_size=my_width*0.66
    font_size=my_width*2.0
    
    font2 = FontProperties()
    font2.set_size('%d'%font_size)
    font2.set_family('sans')
    font2.set_style('normal')
    font2.set_weight('bold')
    plt.tick_params('x', colors='k', size=tick_size)
    plt.tick_params('y', colors='k', size=tick_size)
    plt.rc('xtick', labelsize=lab_size) 
    plt.rc('ytick', labelsize=lab_size)
    

    pb_i = par_dict['pb_i']
    pb_o = par_dict['pb_o']
    plt.set_cmap('coolwarm')
    im=plt.hexbin((mjd/pb_i)%1, mjd,res,size) 
    plt.xlabel("inner phase", fontproperties=font2)
    plt.ylabel("MJD", fontproperties=font2)
    if colorbar:
        cb = plt.colorbar(im)
        cb.set_label(r'$\mu$s', fontproperties=font0)
    plt.gcf().set_size_inches(my_width,my_width*0.77)
    return


def fake_arrows(par_dict,Acols,mjd,unc,ampl):
    A=matrix(par_dict,Acols,mjd)
    amplitude=ampl*np.random.randn(np.shape(A.A)[1])
    my_dict = dict(zip(A.names,amplitude))
    X = []
    Y = []
    U = []
    V = []
    for (i,j,f) in my_dict.keys():
        Y.append(i)
        X.append(j)
        U.append(my_dict[i,j,'cos'])
        if (i,j)==(0,0):
            V.append(0)
        else:
            V.append(my_dict[i,j,'sin'])
    M=np.hypot(V,U)
    class Result(object):
         pass
    pl = Result()
    pl.X = np.array(X)
    pl.Y = np.array(Y)
    pl.U = np.array(U)
    pl.V = np.array(V)
    pl.M = np.array(M)
    
    arrows=np.dot(A.A,amplitude)+unc
    return arrows, amplitude

def get_arrow_length(pl):
    my_pl=np.concatenate((pl.U, pl.V), axis=0)
    my_pl_err = np.concatenate((pl.err_U,pl.err_V), axis=0)
    mu, std = norm.fit(my_pl, q=my_pl_err)
    return std

