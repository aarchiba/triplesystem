
import numpy as np
from numpy.testing import assert_allclose

import threebody
import kepler
import quad_integrate


def setup_solver(shapiro=False,special=False,general=False,roemer=False):
    tol = 1e-16
    initial_values, _ = kepler.kepler_three_body_measurable(
            *(threebody.best_parameters+[0,np.zeros(3),np.zeros(3),0]))
    rhs = quad_integrate.KeplerRHS(special=special, general=general)
    if special or general:
        initial_values = np.concatenate((initial_values,[0]))
    O = quad_integrate.ODE(rhs, initial_values, 0, rtol=tol, atol=tol)
    OD = quad_integrate.ODEDelay(rhs, initial_values, 0, rtol=tol, atol=tol,
            shapiro=shapiro, roemer=roemer)

    return locals()

def test_relativity():
    t_bb = 1.
    S = setup_solver( 
            shapiro=False,
            special=True,
            general=True,
            roemer=True)
    S['OD'].integrate_to(t_bb)

def test_stop():
    for s in range(4):
        yield check_stop, s, False, False, False, True
    yield check_stop, s, True, False, False, True
def check_stop(t_bb, shapiro, special, general, roemer):
    S = setup_solver( 
            shapiro=shapiro,
            special=special,
            general=general,
            roemer=roemer)
    S['OD'].integrate_to(t_bb)
    assert abs(S['OD'].t_bb-t_bb)<1e-10/86400.    

def test_consistency():
    yield check_consistency, False,False,False,True
    yield check_consistency, True,False,False,True
    yield check_consistency, True,True,True,True
def check_consistency(shapiro,special,general,roemer):
    t_bb = 1.
    S = setup_solver( 
            shapiro=shapiro,
            special=special,
            general=general,
            roemer=roemer)
    S['OD'].integrate_to(t_bb)
    S['O'].integrate_to(S['OD'].t_d)
    
    assert_allclose(S['OD'].x, S['O'].x)

def test_roemer():
    t_bb = 1.
    S = setup_solver( 
            shapiro=False,
            special=False,
            general=False,
            roemer=True)
    S['OD'].integrate_to(t_bb)
    assert_allclose(S['OD'].t_d-S['OD'].t_bb,S['OD'].x[2]/86400.)

def test_shapiro_compare():
    t_bb = 1.
    S = setup_solver( 
            shapiro=True,
            special=True,
            general=True,
            roemer=True)
    S['OD'].integrate_to(t_bb)
    x = S['OD'].x
    assert_allclose(threebody.shapiros(x)/86400., 
            quad_integrate.shapiro_delay_l(x))
