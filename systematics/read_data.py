import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from collections import namedtuple
from pylab import rcParams
rcParams['savefig.dpi'] = 144
import pickle
from astropy.io import fits
import os, sys, math, time
import math
from scipy.stats import norm
import matplotlib.mlab as mlab
from scipy import stats
from scipy.stats import kstest
from scipy.stats import norm
from matplotlib.font_manager import FontProperties

def mpu(data):
    limiit=10000
    base_mjd=data['base_mjd']
    c = abs(data['residuals']/data['phase_uncerts'])<limiit
    phase = np.array(data['residuals'][c], dtype=np.float64)
    mjd = np.array(data['times'][c], dtype=np.float64)+base_mjd
    unc = np.array(data['phase_uncerts'][c], dtype=np.float64)
    return mjd, phase, unc

tel_out=namedtuple("tel_transform",
       ["phase", "mjd","unc", "index", "name"])

def tel_transform(data,name,limiit=None):
    '''takes data package and extracts residuals corresponding to particular telescope.
       gives back dictionary of phase and mjd, unc, and buller array of indexes (corresponding indexes from the common data dictionary)
data - pickle or npz package
name - name of the telescope in quotes
limiit - in case you want to exctract only high SN ratio residuals. Use this as an upper limit'''
    if name == 'all':
        tel_res=np.array(data['residuals'])
        tel_unc=np.array(data['phase_uncerts'])
        tel_mjd=np.array(data['times'])
        tel_index= np.ones((len(data['residuals'])), dtype = np.bool)
    else:
        tel_res=np.array(data['residuals'][data['telescopes']==list(data['telescope_list']).index(name)], dtype=np.float64)
        tel_unc=np.array(data['phase_uncerts'][data['telescopes']==list(data['telescope_list']).index(name)], dtype=np.float64)
        tel_mjd=np.array(data['times'][data['telescopes']==list(data['telescope_list']).index(name)], dtype=np.float64)+data['base_mjd']
        tel_index=(data['telescopes']==list(data['telescope_list']).index(name))
    
    if limiit is None:
        limm=10000
    else:
        limm=limiit
    tel_phase=tel_res[abs(tel_res/tel_unc) <limm]
    tel_mjd=tel_mjd[abs(tel_res/tel_unc) <limm]
    tel_unc=tel_unc[abs(tel_res/tel_unc) <limm]
    
    return tel_out(phase=tel_phase, mjd=tel_mjd,unc=tel_unc, index=tel_index, name=name)


import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

mjd_year = 55197
year_mjd = 2010
def day_to_year(mjd):
    return (mjd-mjd_year)/365.2425+year_mjd



err_gr=np.array([129, 199, 132])/255.0
face_gr=np.array([76, 175, 80])/255.0
mec_gr='darkgreen'

err_ye=np.array([255, 238, 88])/255.0
face_ye=np.array([255, 235, 59])/255.0
mec_ye=np.array([245, 127, 23])/255.0

err_bl=np.array([159, 168, 218])/255.0
face_bl=np.array([92, 107, 192])/255.0
mec_bl=np.array([40, 53, 147])/255.0

err_red=np.array([236, 64, 122])/255.0
face_red=np.array([194, 24, 91])/255.0
mec_red=np.array([136, 14, 79])/255.0

err_mag=np.array([175, 122, 197])/255.0
face_mag=np.array([142, 68, 173])/255.0
mec_mag=np.array([74, 35, 90])/255.0

def color_fun(color):

    if color!='g' or color!='y' or color!='b' or color!='r' or color!='m':
        err_clr=color
        face_clr=color
        mec_clr=color

    if color=='g':
        err_clr=err_gr
        face_clr=face_gr
        mec_clr='darkgreen'

    if color=='y':
        err_clr=err_ye
        face_clr=face_ye
        mec_clr=mec_ye

    if color=='b':
        err_clr=err_bl
        face_clr=face_bl
        mec_clr=mec_bl

    if color=='r':
        err_clr=err_red
        face_clr=face_red
        mec_clr=mec_red

    if color=='m':
        err_clr=err_mag
        face_clr=face_mag
        mec_clr=mec_mag
    return err_clr, face_clr, mec_clr


def day_to_tasc(mjd, base_mjd, tasc_o, pb_o):
    zero_tasc=tasc_o+base_mjd# - pb_o
    tasc_time=(mjd-zero_tasc)/pb_o
    return tasc_time


def plot_res_tasc(data, mjd, phase, unc, name, color='b',scl=None):
    base_mjd=data['base_mjd']    
    bp = data["best_parameters"]
    #p_period=1./bp['f0']
    p_period = 0.00273258863228
    pb_i = bp['pb_i']
    pb_o = bp['pb_o']
    tasc_i = bp["tasc_i"]
    tasc_o = bp["tasc_o"]
    microsec=1e6*p_period
    
    if scl is None:
        scale=microsec
    else:
        scale=scl
    
    #a0_format:
    im_width=15
    my_width=15
    mar_size=my_width*0.28
    lab_size=my_width*1.4
    tick_size=my_width*0.66
    font_size=my_width*1.8    
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twiny()
    clr_grey=np.array([170, 183, 184])/255.0
    #for i in range(0,6):
    #    t = pb_o*i+tasc_o+base_mjd
    #    ax1.axvline(t, linestyle='--', color=clr_grey)
        
    for k in range(0,len(color)):
        tel=tel_transform(data,'%s'%name[k])
        err_clr, face_clr, mec_clr=color_fun(color[k])
	tasc_time=day_to_tasc(mjd[tel.index], base_mjd, tasc_o, pb_o)           
        ax1.errorbar(tasc_time, phase[tel.index]*scale, 
                     yerr=unc[tel.index]*scale, 
                     marker="o", 
                     linestyle='none', 
                     mec=mec_clr, 
                     markerfacecolor=face_clr, 
                     color=err_clr, 
                     markersize=mar_size)

    plt.rc('xtick', labelsize=lab_size) 
    plt.rc('ytick', labelsize=lab_size) 

    majorLocator = MultipleLocator(50)
    majorFormatter = FormatStrFormatter('%.0f')
    minorLocator = MultipleLocator(10)
    yearLocator = MultipleLocator(1)

    font0 = FontProperties()
    font0.set_size('%d'%font_size)
    font0.set_family('sans')
    font0.set_style('normal')
    font0.set_weight('bold')

    ax1.set_xlabel('Outer orbital phase',  fontproperties=font0)
    ax1.tick_params('x', colors='k', size=tick_size)
    ax1.tick_params('y', colors='k', size=tick_size)

    ax2.set_xlabel('Years',  fontproperties=font0)
    ax2.tick_params('x', colors='k', size=tick_size)
    ax2.xaxis.set_major_formatter(majorFormatter)
    ax1.tick_params('x', colors='k', size=tick_size)
    if scale==microsec:
        ax1.set_ylabel(r'residuals ($\mu$s)', fontproperties=font0)
    else:
        ax1.set_ylabel(r'residuals', fontproperties=font0)
    ax2.xaxis.set_major_locator(yearLocator)

    ax1.set_xlim(0.1,6.1)

    #d_n, d_x = ax1.get_xlim()
    ax2.set_xlim(day_to_year(base_mjd), day_to_year(base_mjd+2010))

    #d_n, d_x = ax1.get_xlim()
    #ax1.set_xlim(day_to_tasc(d_n, base_mjd, tasc_o, pb_o), day_to_tasc(d_x, base_mjd, tasc_o, pb_o))
    print ax1.get_xlim()
    print ax2.get_xlim()
    
    plt.gcf().set_size_inches(im_width,im_width*4./15.)

    return

def plot_res(data, mjd, phase, unc, name, color='b',scl=None):
    base_mjd=data['base_mjd']    
    bp = data["best_parameters"]
    #p_period=1./bp['f0']
    p_period = 0.00273258863228
    pb_i = bp['pb_i']
    pb_o = bp['pb_o']
    tasc_i = bp["tasc_i"]
    tasc_o = bp["tasc_o"]
    microsec=1e6*p_period
    
    if scl is None:
        scale=microsec
    else:
        scale=scl
    
    #a0_format:
    im_width=15
    my_width=15
    mar_size=my_width*0.28
    lab_size=my_width*1.4
    tick_size=my_width*0.66
    font_size=my_width*1.8    
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twiny()
    clr_grey=np.array([170, 183, 184])/255.0
    for i in range(0,6):
        t = pb_o*i+tasc_o+base_mjd
        ax1.axvline(t, linestyle='--', color=clr_grey)
        
    for k in range(0,len(color)):
        tel=tel_transform(data,'%s'%name[k])
        err_clr, face_clr, mec_clr=color_fun(color[k])           
        ax1.errorbar(mjd[tel.index], phase[tel.index]*scale, 
                     yerr=unc[tel.index]*scale, 
                     marker="o", 
                     linestyle='none', 
                     mec=mec_clr, 
                     markerfacecolor=face_clr, 
                     color=err_clr, 
                     markersize=mar_size)

    plt.rc('xtick', labelsize=lab_size) 
    plt.rc('ytick', labelsize=lab_size) 

    majorLocator = MultipleLocator(50)
    majorFormatter = FormatStrFormatter('%.0f')
    minorLocator = MultipleLocator(10)
    yearLocator = MultipleLocator(1)

    font0 = FontProperties()
    font0.set_size('%d'%font_size)
    font0.set_family('sans')
    font0.set_style('normal')
    font0.set_weight('bold')

    ax1.set_xlabel('MJD',  fontproperties=font0)
    ax1.tick_params('x', colors='k', size=tick_size)
    ax1.tick_params('y', colors='k', size=tick_size)

    ax2.set_xlabel('Years',  fontproperties=font0)
    ax2.tick_params('x', colors='k', size=tick_size)
    ax2.xaxis.set_major_formatter(majorFormatter)
    ax1.tick_params('x', colors='k', size=tick_size)
    if scale==microsec:
        ax1.set_ylabel(r'residuals ($\mu$s)', fontproperties=font0)
    else:
        ax1.set_ylabel(r'residuals', fontproperties=font0)
    ax2.xaxis.set_major_locator(yearLocator)

    ax1.set_xlim(base_mjd,base_mjd+2010)

    d_n, d_x = ax1.get_xlim()
    ax2.set_xlim(day_to_year(d_n), day_to_year(d_x))
    print ax1.get_xlim()
    print ax2.get_xlim()
    
    plt.gcf().set_size_inches(im_width,im_width*4./15.)

    return

#Function to fit normal distribution to it and plot it
def fit_plot(emac, number, name, color, numplots, limm):
    err_clr, face_clr, mec_clr=color_fun(color)
    mu, std = norm.fit(emac)
    ax=plt.subplot(2, int(numplots/2), number)
    ax.hist(emac, bins=100, normed=True, alpha=0.8, color=err_clr)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, 'k', linewidth=2)
    ax.set_xlabel(r'$\mu$ = %.4f,  $\sigma$ = %.4f'%(mu, std), fontsize=20)
    ax.set_ylabel('res/unc', fontsize=20)
    ax.set_xlim(-limm,limm)
    if numplots == 4:
        if number%2:
            xx=0.13
        else:
            xx=0.55
        if (number >2.5):
            yy=0.45
        else:
            yy=0.87
    if numplots == 6:
	if number%3:
            if number%2:
                 xx=0.13
            else:
                 xx=0.40
        else:
            xx=0.70

        if (number >3.5):
            yy=0.45
	    if number==4:
                 xx=0.13
            if number==5:
                 xx=0.40
        else:
            yy=0.87
	
    plt.figtext(xx,yy, '%s'%name, fontsize=20, color=mec_clr)
    return std


#Function to fit normal distribution to it and plot it
def fit_single_plot(emac, name, color, limm):
    err_clr, face_clr, mec_clr=color_fun(color)
    mu, std = norm.fit(emac)
    plt.hist(emac, bins=100, normed=True, alpha=0.6, color=err_clr)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    plt.xlabel(r'$\mu$ = %.4f,  $\sigma$ = %.4f'%(mu, std), fontsize=20)
    plt.ylabel('res/unc', fontsize=20)
    plt.xlim(-limm,limm)
    xx=0.13
    yy=0.87
    plt.figtext(xx,yy, '%s'%name, fontsize=20, color=mec_clr)
    return std



def plot_hex(par_dict, mjd,res, size, colorbar=True, vmin=None, vmax=None):
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
    im=plt.hexbin((mjd/pb_i)%1, mjd,res,size,linewidths=0.1, vmin=vmin, vmax=vmax) 
    plt.xlabel("inner phase", fontproperties=font2)
    plt.ylabel("MJD", fontproperties=font2)
    if colorbar:
        cb = plt.colorbar(im)
        cb.set_label('ns', fontproperties=font2)
    plt.gcf().set_size_inches(my_width*0.77,my_width)
    return



def plot_unc(my_data, phase, unc, name, color='b',scl=None):

    #a0_format:
    im_width=15
    my_width=15
    mar_size=my_width*0.28
    lab_size=my_width*1.4
    tick_size=my_width*0.66
    font_size=my_width*1.8    

    if scl == None:
    	scale=1
    else:
    	scale=scl

    for k in range(0,len(color)):
        tel=tel_transform(data,'%s'%name[k])
        err_clr, face_clr, mec_clr=color_fun(color[k])

        ax1.errorbar(res[tel.index]*scale, unc[tel.index]*scale,
                     yerr=unc[tel.index]*scale,
                     marker="o",
                     linestyle='none',
                     mec=mec_clr,
                     markerfacecolor=face_clr,
                     color=err_clr,
                     markersize=mar_size)


    my_grey=np.array([213, 219, 219])/255.0
    plot_unc(my_data['residuals'],my_data['phase_uncerts'],scl, my_grey,'lightgrey',my_grey,'with outliers')
    
    plt.xscale('log')
    plt.xlim(7e-2,3.5)
    plt.ylim(-30,30)
    
    lgd = plt.legend(numpoints=3, loc='upper left', bbox_to_anchor=(0.125, 1), fontsize=7)
    plt.ylabel(r'residuals ($\mu$s)')
    plt.xlabel(r'Unc. ($\mu$s)')
    
    plt.gcf().set_size_inches(im_width,im_width*0.6) 
