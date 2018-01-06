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


#************************************************************************************FIT**FUNCTIONS*******************************
#-----function-1 --- make a matrix
#---------------------------------

def matrix(par_dict, ll, mjd):
    '''Creates matrix [ll x len(mjd)] of cos in sin with k phi_i + l phi_o phases.
    k =[-ll,+ll]; l=[0,ll]
    Returns: matrix, names (cos or sin components) and dictionarry that combines them.

par_dict - dictionary with pulsar parameters (phi and t_asc)
Acols - max number of inner orbital frequencies to calculate
mjd - TOAs'''

    Adict = {}
    base_mjd=par_dict['base_mjd']
    pb_i = par_dict['pb_i']
    pb_o = par_dict['pb_o']
    phi_i=(2.0*math.pi)/pb_i
    phi_o=(2.0*math.pi)/pb_o
    tasc_i = par_dict["tasc_i"]+base_mjd
    tasc_o = par_dict["tasc_o"]+base_mjd
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

#-----------------------------------------------
#-----function-2 --- fit A using matrix function
#-----------------------------------------------

def linear_least_squares_cov(par_dict, Acols, mjd, phase, unc):
    """Solve a linear least-squares problem and return uncertainties.
    """

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

#----------------------------------------------------------
#-function-------general-lstsq-fit-with-uncertainties------
#----------------------------------------------------------
def lstsq_with_errors(A, b, uncerts=None):
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
    if len(A.shape) != 2:
        raise ValueError
    if uncerts is None:
        Au = A
        bu = b
    else:
        Au = A/uncerts[:, None]
        bu = b/uncerts
    Ascales = np.sqrt(np.sum(Au**2, axis=0))
    # Ascales = np.ones(A.shape[1])
    if np.any(Ascales == 0):
        raise ValueError("zero column (%s) in A" % np.where(Ascales == 0))
    As = Au/Ascales[None, :]
    db = bu
    xs = None
    best_chi2 = np.inf
    best_x = None
    for i in range(5):   # Slightly improve quality of fit
        dxs, res, rk, s = scipy.linalg.lstsq(As, db)
        if rk != A.shape[1]:
            raise ValueError("Condition number still too bad; "
                             "singular values are %s and rank "
                             "should be %d"
                             % (s, A.shape[1]))
        if xs is None:
            xs = dxs
        else:
            xs += dxs
        db = bu - np.dot(As, xs)
        chi2 = np.sum(db**2)
        if chi2 < best_chi2:
            best_chi2 = chi2
            best_x = xs
        #debug("Residual chi-squared: %s", np.sum(db**2))
    x = best_x/Ascales    # FIXME: test for multiple b

    class Result:
        pass
    r = Result()
    r.x = x
    r.residuals = b-np.dot(A, x)
    if uncerts is None:
        r.residuals_scaled = r.residuals
    else:
        r.residuals_scaled = r.residuals/uncerts
    r.singular_values = s
    r.chi2 = best_chi2
    bias_corr = A.shape[0]/float(A.shape[0]-A.shape[1])
    r.reduced_chi2 = bias_corr*r.chi2/A.shape[0]
    Atas = np.dot(As.T, As)
    covs = scipy.linalg.pinv(Atas)
    r.cov = covs/Ascales[:, None]/Ascales[None, :]
    r.uncerts = np.sqrt(np.diag(r.cov))
    r.corr = r.cov/r.uncerts[:, None]/r.uncerts[None, :]
    return r


################################################################################################
####-------------------Derivatives---------------------------------------------------------#####
################################################################################################

def der_fit(data,phase,unc):
    """Does additional fit of residuals (phase) for all linear parameters using 'derivatives' (responce
       of residuals to each parameter) provided by timing fit.
       Returns corrected array of residuals
data -- data package ectracted form pickle (contains phase, mjd, unc, all fit parameters and corresponing
derivatives
phase, unc -- residuals you want to refit and its uncertaities.
"""


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
    #print np.shape(cols)

    del cols[cols.index(par)]
    A = np.array([d[c] for c in cols]).T

    #print A.shape
    b=d[par]

    r = lstsq_with_errors(A, b, unc)
    der_par=(b-np.dot(A,r.x))
    return der_par


def der_unc(data):
    """Does additional fit of residuals (phase) for all linear parameters using 'derivatives' (responce
       of residuals to each parameter) provided by timing fit.
       Returns corrected array of residuals
data -- data package ectracted form pickle (contains phase, mjd, unc, all fit parameters and corresponing
derivatives
phase, unc -- residuals you want to refit and its uncertaities.
"""

    phase=data['residuals']
    unc=data['phase_uncerts']
    d=np.array(data['derivatives'])[np.newaxis][0]
    cols= sorted(d.keys())

    A = np.array([d[c] for c in cols]).T
    b=phase
    r = lstsq_with_errors(A, b, unc)
    
    for (i,c) in enumerate(cols):
        if c in data['best_parameters']:
            v = data['best_parameters'][c]
        elif c in data['linear_parameters']:
            v = data['linear_parameters'].index(c)
        else:
            v = 0
        print i, c, v-r.x[i], "+/-", r.uncerts[i], (r.x/r.uncerts)[i], "sigma"
    
    return



#************************************************************************************************************************ARROWS_OBJECT_AND_FUNCTIONS*******************

#-Object-----------------------------------
#combines arrow coefficients in a nice way:
#------------------------------------------
class Coefficients(object):
    def __init__(self, X, Y, U, V, err_U=None, err_V=None, err_X=None, err_Y=None, err_ix=None):
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.U = np.array(U)
        self.V = np.array(V)
        self.err_U = err_U
        self.err_V = err_V
        self.err_X = err_X
        self.err_Y = err_Y
        self.err_ix = err_ix
        self.M = np.hypot(self.U, self.V)
        if len(X)!=len(Y):
            raise ValueError("Arrays X and Y not the same length")
        if len(U)!=len(V):
            raise ValueError("Arrays U and V not the same length")
        if len(U)!=len(X):
            raise ValueError("Arrays X and U not the same length")
        if err_U !=None:
            if len(U)!=len(err_U):
                raise ValueError("Arrays U and err_U not the same length")
            if len(V)!=len(err_V):
                raise ValueError("Arrays V and err_V not the same length")

    def __repr__(self):
        return "<Coefficients length %d>" % len(self.X)

    def __add__(self, other):
	if len(self.X)!=len(other.X):
            raise ValueError("Arrays are not the same length")
	
	res_U = other.U + self.U
	res_V = other.V + self.V
	res=Coefficients(self.X, self.Y, res_U, res_V)
	return res

    def __sub__(self, other):
	if len(self.X)!=len(other.X):
            raise ValueError("Arrays are not the same length")
	
	subs_U = other.U - self.U
	subs_V = other.V - self.V
	subs=Coefficients(self.X, self.Y, subs_U, subs_V)
	return subs


#---------------------------------------------------
#--function-3---create_plotable_arrows (arrow array)
#---------------------------------------------------

def ar_coeff(par_dict, Acols, mjd, phase, unc, chol=True):
    '''Calculates arrow coefficients and values for a given data set.
       Returns: object with arrows (coordinates, directions and lengths as well as 1-sigma errors) 
par_dict - dictionary with pulsar parameters
Acols - max number of inner orbital frequencies to calculate
mjd, phase, unc - parameters of data'''
    if chol==True:
    	ang = np.linspace(0,2*np.pi,100)
	circle = np.array([np.cos(ang),np.sin(ang)])
	err_X = []
        err_Y = []
        err_ix = []
        err_U = []
        err_V = []
    else:
	err_X=None
	err_Y=None
	err_ix=None
        err_U=None
        err_V=None

    fit = linear_least_squares_cov(par_dict, Acols, mjd, phase, unc)
    result = dict(zip(fit.names,fit.x))
    X = []
    Y = []
    U = []
    V = []
   
    for k in range(len(result.keys())):
        aa=result.keys()[k]
        if aa[2]=='cos':
            Y.append(aa[0])
            X.append(aa[1])
            U.append(result[aa[0],aa[1],'cos'])
	    if chol==True:
                err_ix.append((aa[0],aa[1]))
            if aa[0]==0 and aa[1]==0:
                V.append(0)
		if chol==True:
	            err_U.append(0)
                    err_V.append(0)
                    err_X.append(0)
                    err_Y.append(0)
            else:
                V.append(result[aa[0],aa[1],'sin'])
		if chol==True:
                    st=fit.names.index((aa[0], aa[1], 'cos'))
                    fin=fit.names.index((aa[0], aa[1], 'sin'))+1
                    err_MM=fit.cov[st:fin,st:fin]
                    err_U.append(np.sqrt(np.diag(err_MM))[0])
                    err_V.append(np.sqrt(np.diag(err_MM))[1])
                    assert err_MM.shape==(2,2)
                    L = scipy.linalg.cholesky(err_MM)#, lower=True)
                    Lcircle = np.dot(L,circle)
                    # print Lcircle[0][7]
                    err_X.append(Lcircle[0])
                    err_Y.append(Lcircle[1])
           
    coeff=Coefficients(X=X, Y=Y, U=U, V=V, err_U=err_U, err_V=err_V, err_X=err_X, err_Y=err_Y, err_ix=err_ix)
    return coeff

#--------------------------------------------------------------
#-----function-5 --- makes an arrow plot from arrow coeffcients
#--------------------------------------------------------------

font0 = FontProperties()
font0.set_size('10')
font0.set_weight('normal')
font0.set_family('serif')
font0.set_style('normal')

err_red=np.array([191, 54, 12])/255.0
err_bl=np.array([21, 101, 192])/255.0


def coeff_plot(coeff, scale=None, plot_unc=True, color=err_bl, units='', lentext=True):
    '''Makes arrow plot for a given (arrow_coeff) object.
       Takes: (arrow_coeff) object. 
       Returns: value of the biggest arrow in input units and draws plot
pl - object with arrow coordinates, directions and lenghts (input your units)'''
    if scale is None:
        ar_scale=np.amax(np.hypot(coeff.V,coeff.U))
    else:
        ar_scale=scale
    if plot_unc:
        for k in range(0,len(coeff.err_X)):
            i,j = coeff.err_ix[k]
            plt.plot(coeff.err_X[k]/ar_scale+j,coeff.err_Y[k]/ar_scale+i, color=color, lw=1, alpha=0.3)
    Q = plt.quiver(coeff.X, coeff.Y, coeff.U, coeff.V, units='x', scale=ar_scale, color=color)
    Acols=np.amax(coeff.X)+1
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

#--------------------------------------------------------
#-----function-5 --- makes an arrow plot from res and mjd
#--------------------------------------------------------

def draw_plot(par_dict, Acols, mjd, phase, unc, scale=None, plot_unc=True, color=err_bl, units='', lentext=True):
    '''Makes arrow plot for a given data set.
       Returns: value of the biggest arrow and plot
par_dict - dictionary with pulsar parameters
Acols - max number of inner orbital frequencies to calculate
mjd, phase, unc - parameters of data (in units you want to get your arrow lenghts in)'''
    coeff=ar_coeff(par_dict, Acols, mjd, phase, unc, chol=True)
    return coeff_plot(coeff,scale=scale, plot_unc=plot_unc, color=color, units=units, lentext=lentext)

#--------------------------------------------------------
#-----function-6 --- makes an arrow plot from data file
#--------------------------------------------------------

def draw_ar_plot_from_data(data,Acols,scale=None, plot_unc=True, color=err_bl, units='', lentext=True, scl_s=None):
    base_mjd=data['base_mjd']
    par_dict=data['best_parameters']
    par_dict['base_mjd']=base_mjd
    p_period = data['f0']**(-1)

    if scl_s == 'ns':
        scl=p_period*1e9
    else:
        if scl_s == 'mus':
            scl=p_period*1e6
        else:
            scl=1.
 
    new_phase = np.array(data['residuals'], dtype=np.float64)
    new_times = np.array(data['times'], dtype=np.float64)+base_mjd
    new_unc = np.array(data['phase_uncerts'], dtype=np.float64)
    return draw_plot(par_dict, Acols, new_times, new_phase*scl, new_unc*scl, scale=scale, plot_unc=plot_unc, color=color, units=units, lentext=lentext)

#-----------------------------------------------------
#-----function------ gives parameters of a given arrow
#-----------------------------------------------------

def arrow_extract(coeff,k,j):
    '''Takes (arrow_coeff) object and frequecncies of the arrow you want to get info about.
       Returns len=3 array: [projection_X, projection_Y, lenght].
coeff - object with arrow coordinates, directions and lenghts
k - inner orbit frequency
j - outer orbit frequency''' 
    for i in range(0,len(coeff.X)):
            if coeff.X[i]==k and coeff.Y[i]==j:
                ll=np.array([coeff.U[i], coeff.V[i], coeff.M[i]])
    return ll




#*******************************************************************************************************************************UPPER_LIMIT_TEST_FUNCTIONS*********

####################################################################################
###--------------------------------Fake-arrows-----------------------------------###
####################################################################################

#-Function------------------------------
#-makes grid and values for arrow coeff:
#---------------------------------------
def generate_xy(x_range, y_range):
    X=[]
    Y=[]
    for i in range(x_range[0], x_range[1]+1):
        for j in range(y_range[0],y_range[1]+1):
            if j<=0 and i<0:
                v=1
            else:
                X.append(i)
                Y.append(j)
    X=np.array(X)
    Y=np.array(Y)
    #print len(X)
    rand_array=np.random.randn(len(X)*2)
    U=rand_array[0:len(X)]
    V=rand_array[len(X):len(X)*2]
    for k in range(0,len(X)):
    	if X[k]==0 and Y[k]==0:
            V[k]=0
    return X, Y, U, V

#--function---------------------------------------------------
#--creates fake arrow coeff for a standard grid used in the fit
#--and makes a fake arrow array and for this coeff.------------
#--------------------------------------------------------------
def fake_arrow_coeff(par_dict,Acols,mjd,ampl):
    A=matrix(par_dict,Acols,mjd)
    amplitude=ampl*np.random.randn(np.shape(A.A)[1])
    assert len(amplitude)==len(A.names)
    my_dict = dict(zip(A.names,amplitude))
    X = []
    Y = []
    U = []
    V = []
    for k in range(len(my_dict.keys())):
         aa=my_dict.keys()[k]
         if aa[2]=='cos':
             Y.append(aa[0])
             X.append(aa[1])
             U.append(my_dict[aa[0],aa[1],'cos'])
             if aa[0]==0 and aa[1]==0:
                 V.append(0)
             else:
                 V.append(my_dict[aa[0],aa[1],'sin'])
 
    coeff = Coefficients(X=X, Y=Y, U=U, V=V)
    return coeff


#-Function------------------------------------------------------------------------
#--makes fake data set with ejceted arrows corresponding to coeff object provided:
#---------------------------------------------------------------------------------
def fake_arrow_array(coeff,par_dict,mjd):
    ll=len(coeff.X)
    Adict = {}
    base_mjd=par_dict['base_mjd']
    pb_i = par_dict['pb_i']
    pb_o = par_dict['pb_o']
    phi_i=(2.0*math.pi)/pb_i
    phi_o=(2.0*math.pi)/pb_o
    tasc_i = par_dict["tasc_i"]+base_mjd
    tasc_o = par_dict["tasc_o"]+base_mjd
    
    for i in range(ll):
        Adict[coeff.X[i],coeff.Y[i],'cos'] = coeff.U[i]*np.cos(coeff.Y[i]*phi_o*(mjd-tasc_o)+coeff.X[i]*phi_i*(mjd-tasc_i))
        Adict[coeff.X[i],coeff.Y[i],'sin'] = coeff.V[i]*np.sin(coeff.Y[i]*phi_o*(mjd-tasc_o)+coeff.X[i]*phi_i*(mjd-tasc_i))
    names = sorted(Adict.keys())
    A = np.array([Adict[n] for n in names]).T
    i_vector=np.zeros((np.shape(A)[1]))
    i_vector=i_vector+1.0
    
    arrows=np.dot(A,i_vector)
    
    return arrows

#-function-----------------------
#--gives-std of the arrow lengths:
#--------------------------------
def std_lengths(coeff, dof):
    my_coeff=np.concatenate((coeff.U, coeff.V), axis=0)
    #my_pl_err = np.concatenate((pl.err_U,pl.err_V), axis=0)
    std = np.std(my_coeff, ddof=dof)
    #mu, std = norm.fit(my_pl, q=my_pl_err)
    return std

#-function---------------------------------
#-calculates sqrt(sum(X^2)) of arrow coeff.
#------------------------------------------
def sqr_lengths(coeff):
    my_coeff=np.concatenate((coeff.U, coeff.V), axis=0)
    res=np.sqrt(np.sum(my_coeff**2.0)/(len(my_coeff)))
    return res



#-function------------------------------------------------------------------
#-makes-big-array-of-fake-arrow-coeff---and--calculates derivatives for them
#-combines it all in pickle-------------------------------------------------
#---------------------------------------------------------------------------
def make_der_array(my_data, my_ampl, ar_len, filename):
    import numpy as np
    import pickle
   
    base_mjd=my_data['base_mjd']
    par_dict=my_data['best_parameters']
    par_dict['base_mjd']=base_mjd
    
    new_phase = np.array(my_data['residuals'], dtype=np.float64)
    new_times = np.array(my_data['times'], dtype=np.float64)+base_mjd
    new_unc = np.array(my_data['phase_uncerts'], dtype=np.float64)
    
    in_final=np.empty(ar_len, dtype=object)
    der_final=np.empty(ar_len, dtype=object)
    
    for i in range(0, len(in_final)):
        init_cf=fake_arrow_coeff(par_dict,4,new_times,my_ampl)
        
        my_arrows=fake_arrow_array(init_cf,par_dict,new_times)
        
        in_cf=ar_coeff(par_dict, 4, new_times, my_arrows, new_unc, chol=False)
        
        in_final[i] = in_cf
        
        new_arrows=der_fit(my_data,my_arrows,new_unc)
        
        der_cf=ar_coeff(par_dict, 4, new_times, new_arrows, new_unc, chol=False)
        
        der_final[i] = der_cf
    
        if np.mod(i,100)==0:
            outf = open('%s_progress.txt'%filename,'w+')
            outf.write('%d\n'%i)
            outf.close()
            print i
        if np.mod(i+1,10000)==0:
            #Pickle it:
            cf_10000 = {
                'init_coeff': in_final[i-9999:i+1],
                'der_coeff': der_final[i-9999:i+1],
                'sigma': my_ampl
            }
            
            with open('%s_%f_%d.pickle'%(filename,my_ampl,i+1), 'wb') as g:
                # Pickle the 'data' dictionary using the highest protocol available.
                pickle.dump(cf_10000, g, pickle.HIGHEST_PROTOCOL)
                
                
    import pickle
    
    der_final_ar = {
        'init_coeff': in_final,
        'der_coeff': der_final,
        'sigma': my_ampl
    }
    
    with open('%s_%3.2f_final.pickle'%(filename,my_ampl), 'wb') as e:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(der_final_ar, e, pickle.HIGHEST_PROTOCOL)
    return in_final, der_final, my_ampl



#-Function-------------------------------------------------------------------------------
#-extracts upper limit on delta base on real data and pre-prepared derivatives collection
#----------------------------------------------------------------------------------------
def upper_limit(my_data, my_pickle, scl_s='ns', scl=None):
    import numpy as np
    import pickle
    import scipy.linalg
    from scipy import stats
    from scipy.stats import norm
    from scipy import optimize

    base_mjd=my_data['base_mjd']
    par_dict=my_data['best_parameters']
    par_dict['base_mjd']=base_mjd
    
    deltaa=my_data['best_parameters']['delta']

    p_period = my_data['f0']**(-1)
    
    limiit=10000
    c = abs(my_data['residuals']/my_data['phase_uncerts'])<limiit
    new_phase = np.array(my_data['residuals'][c], dtype=np.float64)
    new_times = np.array(my_data['times'][c], dtype=np.float64)+base_mjd
    new_unc = np.array(my_data['phase_uncerts'][c], dtype=np.float64)
    
    if scl_s == 'ns':
        scl=p_period*1e9
    else:
        if scl_s == 'mus':
            scl=p_period*1e6
	else:
	    if scl == None:
		scl=1.
	    else:
                scl=scl
    
    #Do additional derivatives    
    better_phase=der_fit(my_data,new_phase,new_unc)
    delta_only=der_of_par(my_data,'delta',new_unc)

    #Caclulate sqrt(sum(X^2)) of the arrow coefficient from the real data fit
    my_coeff=ar_coeff(par_dict, 4, new_times, better_phase*scl, new_unc*scl)
    my_F=sqr_lengths(my_coeff)
    print 'sqrt(sum(X^2)) in %s:'%scl_s, my_F

    #Get the value of delta from the arrow fit:
    signal_delta=draw_plot(par_dict, 4, new_times, delta_only*deltaa*scl, new_unc*scl*1e-12, color='red')
    print 'signal of Delta in %s'%scl_s, signal_delta, '; the fit value of Delta', deltaa
    
    #Unpickle array with der_coeff collection:
    with open('%s'%my_pickle, 'rb') as f:
        array_pickle = pickle.load(f)

    my_in=array_pickle['init_coeff']
    my_der=array_pickle['der_coeff']
    my_sigma_in=array_pickle['sigma']

    my_sigma=my_sigma_in*scl#where my_sigma_in is the amplitude of the generated systematic what I input (in pulsar rotations)
    #my_sigma is this amplitude in ns
    real_F=my_F#real_F root mean squared of observed arrow coefficients (as a result of the fit of the real data). in ns
    my_der=my_der# collection of derivatives substructed coeff. I builded up.

    print 'len of der array:', len(my_in)
    

    #Calculate resulted mean sigma:
    F_list=[]
    sigma_list=[]

    for i in range(0,len(my_der)):
        F=sqr_lengths(my_der[i])*scl
        F_list.append(F)
        sig=(my_sigma*real_F)/F
        sigma_list.append(sig)
        #print 'F', i, '=', F, 'mus', (my_sigma*real_F)/F

    F_array=np.array(F_list, dtype=np.float64)
    sigma_array=np.array(sigma_list, dtype=np.float64)

    print 'mean of resulted sigma (%s):'%scl_s, np.mean(sigma_array), '; its std:', np.std(sigma_array)
    mean_sigma=np.mean(sigma_array)

    #Calculate the survival function:
    my_sf=norm.sf(signal_delta, 0, sigma_array).mean()*2./(2*norm.sf(1))

    print 'sf(real_Delta):', my_sf
    
    #root finder:

    def my_fun_1(a):
        return norm.sf(a, 0, sigma_array).mean()*2. - (2*norm.sf(1))
    def my_fun_2(a):
        return norm.sf(a, 0, sigma_array).mean()*2. - (2*norm.sf(2))
    def my_fun_3(a):
        return norm.sf(a, 0, sigma_array).mean()*2. - (2*norm.sf(3))


    my_sol_1 = optimize.root(my_fun_1, [mean_sigma], method='hybr')
    my_sol_2 = optimize.root(my_fun_2, [mean_sigma*2.], method='hybr')
    my_sol_3 = optimize.root(my_fun_3, [mean_sigma*3.], method='hybr')
    print '1_sigma', my_sol_1.x[0], '%s'%scl_s
    print '2_sigma', my_sol_2.x[0], '%s'%scl_s
    print '3_sigma', my_sol_3.x[0], '%s'%scl_s
    
    #The upper limit:
    delta_value=deltaa*my_sol_1.x[0]/signal_delta
    print '1-sigma upper limit on Delta:', delta_value
    return delta_value


