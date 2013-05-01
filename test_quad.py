import warnings

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
    for s in range(1,5):
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
    assert_allclose(S['OD'].t_bb-S['OD'].t_d,S['OD'].x[2]/86400.)

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

def test_inverses():
    yield (check_inverse, 
            lambda args: tuple(kepler.kepler_2d(*args)[0])+(kepler.mass(args[0],args[1]),),
            lambda state: kepler.inverse_kepler_2d(state[:4],state[4]),
            (1., 1., 0.1, 0.2, 0.4))
    yield (check_inverse, 
            lambda args: tuple(kepler.kepler_3d(*args)[0])+(kepler.mass(args[0],args[1]),),
            lambda state: kepler.inverse_kepler_3d(state[:6],state[6]),
            (1., 1., 0.1, 0.2, 0.3, 0.5, 0.4))
    yield (check_inverse, 
            lambda state: kepler.inverse_kepler_two_body(state),
            lambda args: kepler.kepler_two_body(*args)[0],
            kepler.kepler_two_body(1., 1., 0.1, 0.2, 0.3, 0.5, 0.4, 
                [0,0,0], [0,0,0], 1.7)[0])
    yield (check_inverse, 
            lambda state: kepler.inverse_kepler_three_body(state),
            lambda args: kepler.kepler_three_body(*args)[0],
            kepler.kepler_three_body(
                1., 1.1, 0.1, 0.2, 0.3, 0.5, 0.4, -0.3,
                100., 327., 0.01, 0.02, 0.13, 0.15, -0.13,
                [0,0,0], [0,0,0])[0])
    t = 5.
    yield (check_inverse, 
            lambda state: kepler.inverse_kepler_three_body_measurable(state,t),
            lambda args: kepler.kepler_three_body_measurable(*(args+(t,)))[0],
            kepler.kepler_three_body_measurable(
                1., 1.1, 0.1, 0.2, -0.3,
                0.8, 0.1,
                100., 327., 0.01, 0.02, -0.13,
                70., 0.25,
                0.18, [0,0,0], [0,0,0],
                t)[0])
def check_inverse(f, finv, val):
    print f(val)
    assert_allclose(finv(f(val)),val)

def test_toolong():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        kepler.inverse_kepler_three_body_measurable(
                np.array([ 113.53708453224969332,
                    21.620509479020036281,
                    17.618429094116024203,
                    6.7666276938390117479,
                    1.7015022642930845986,
                    1.386924222905377535,
                    1.4415130881788671413,
                    113.52081280307139366,
                    33.987002810439491896,
                    27.704269571410936379,
                    -54.684761698699944077,
                    1.6572798256903455982,
                    1.3508573728039836581,
                    0.19805927770897602258,
                    -452.54426924620975115,
                    -92.132527112439376538,
                    -75.082339166353676774,
                    2.6174089698788773894,
                    -6.7607943312246314704,
                    -5.5108317714803343179,
                    0.41133885939089165351,
                    0])
                    , 0)
