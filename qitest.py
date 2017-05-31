import numpy as np
from numpy.random import normal
import numpy.testing

import matplotlib.pyplot as plt

import kepler
import quad_integrate

print 

print """Testing long double accuracy is not lost

Each line contains a number close to one and 
that number minus one. The difference is chosen
to come out to zero if the number has been truncated
to double accuracy.

"""
t = np.float128(1)
eps = 2.**(-60)
print "epsilon:", eps
t += eps
print "python float:", 1., (1.+eps)-1
print "np.float128:", t, t-1
O = quad_integrate.ODE(quad_integrate.PyRHS(None),
        np.array([t,2,3],dtype=np.float128), t, 0)
print "Created ODE"
tout = O.t
print "via ODE", tout, tout-1, type(tout)#, tout.dtype
print "x:", O.x, O.x[0]-1
print

def rhs(x, t):
    d = np.zeros_like(x)
    d[0] = x[1]
    d[1] = -x[0]
    return d
ts = []
xs = []
O = quad_integrate.ODE(quad_integrate.PyRHS(rhs),
        [1,0],0)
O = quad_integrate.ODE(quad_integrate.HarmonicRHS(),
        [1,0],0)
for t in np.linspace(0,2*np.pi,100):
    O.integrate_to(t)
    ts.append(O.t)
    xs.append(O.x)
print "n:", O.n_evaluations

ts = np.array(ts)
xs = np.array(xs)
print ts.dtype, xs.dtype
plt.plot(ts,xs[:,0])
plt.plot(ts,xs[:,1])
#plt.gca().set_yscale('symlog')
plt.show()



state = normal(size=14)
state[6::7] = np.exp(state[6::7])

numpy.testing.assert_allclose(
        kepler.rhs_three_body(0,state),
        quad_integrate.KeplerRHS().evaluate(state.astype(np.float128), 0))


state = normal(size=21)
state[6::7] = np.exp(state[6::7])

numpy.testing.assert_allclose(
        kepler.rhs_three_body(0,state),
        quad_integrate.KeplerRHS().evaluate(state.astype(np.float128), 0))

