There are a few functions which do some systematics search

##############
1)plotter_fun.py
Consist of functions plotter and arrow_extract:

-----
plotter(par_dict, Acols, mjd, phase, unc):
    '''Calculates arrow coefficients and values for a given data set.
       Returns: object with arrows (coordinates, directions and lengths as well as 1-sigma errors) 
par_dict - dictionary with pulsar parameters
Acols - max number of inner orbital frequencies to calculate
mjd, phase, unc - parameters of data''' 

-----
arrow_extract(pl,k,j):
    '''Takes (arrow_coeff) object and frequecncies of the arrow you want to get info about.
    Returns len=3 array: [projection_X, projection_Y, lenght].
pl - object with arrow coordinates, directions and lenghts
k - inner orbit frequency
j - outer orbit frequency'''
-----

How to call:
import plotter_fun.py as pltr

my_arrow_object=pltr.plotter(par_dict, Acols, mjd, phase, unc)
my_arrow_info=plt.arrow_exctract(pl,k,j)

###################3
2) arrow_plot_fun.py
Consist of functions draw_plot and coeff_plot

-----
draw_plot(par_dict, Acols, mjd, phase, unc, scale=None):
    '''Makes arrow plot for a given data set.
       Returns: value of the biggest arrow and plot
par_dict - dictionary with pulsar parameters
Acols - max number of inner orbital frequencies to calculate
mjd, phase, unc - parameters of data'''

-----
coeff_plot(pl, scale=None):
    '''Makes arrow plot for a given (arrow_coeff) object.
       Takes: (arrow_coeff) object. 
       Returns: value of the biggest arrow and plot
pl - object with arrow coordinates, directions and lenghts''' 
-----

How to call:
import arrow_plot_fun.py as ar_pl
my_value=ar_pl.draw_plot(par_dict, Acols, mjd, phase, unc)
my_value = ar_pl.coeff_plot(pl)

##########################
xx) matrix_fun.py
Consist of function matrix
Will create a matrix of harmonics to use in lsqr fit using parameters of the system, TOAs, and given maximum number of garmonics.

-----
matrix(par_dict,Acols,mjd)
par_dict - dictionary with system parameters.
Acols - max number of inner orbital frequencies to calculate
mjd - TOAs'''
-----

How to call:
import matrix_fun as mtrx
my_matrix=mtrx.matrix(par_dict,ll,mjd)




etc...
