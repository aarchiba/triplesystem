# distutils: language = c++
# distutils: include_dirs = include
# distutils: libraries = quadmath
# distutils: depends = extra.hpp quad_defs.hpp ppn.hpp
# distutils: extra_compile_args = -march=native

import sys

from libcpp.vector cimport vector

cimport cython

import numpy as np
cimport numpy as np


DTYPE = np.float128
ctypedef np.npy_float128 DTYPE_t

# Magic to make __float128 appear
cdef extern from *:
    ctypedef long double quad # this is only for cython, where treating real quads as long doubles is fine
ctypedef long double longdouble
ctypedef vector[longdouble] vectl
ctypedef vector[quad] vectq


cdef extern from "extra.hpp":
    enum quad128values:
        FLT128_MAX
    void vectq_set(vectq*v, size_t i, quad e)
    void vectl_set(vectl*v, size_t i, longdouble e)
    ctypedef void (*callback_function)(
            vectq*x, vectq*dxdt, quad t, void* arg) except *
    longdouble quad_to_longdouble(quad q)

    cdef cppclass CRHS[num]:
        CRHS(void (*cb)(vector[num]*,vector[num]*,num,void*), void*arg) except +
        void evaluate(vector[num] &x, vector[num] &dxdt, num t) except +
    cdef cppclass cKeplerRHS[num]:
        num gamma
        num beta
        cKeplerRHS(int special, int general, long long&evals, num delta,
                   int ppn_motion, num gamma, num beta,
                   num Gamma01, num Gamma02, num Gamma12,
                   num Theta01, num Theta02, num Theta12,
                   num Gamma011, num Gamma012, num Gamma022,
                   num Gamma100, num Gamma102, num Gamma122,
                   num Gamma200, num Gamma201, num Gamma211,
                   int matrix_mode, num c_scale) except +
        void evaluate(vector[num] *x, vector[num] *dxdt, num t) except +
        long long n_evaluations()
    cdef longdouble shapiro_delay(vectl&x, longdouble gamma)
    cdef quad shapiro_delay(vectq&x, quad gamma)

    cdef cppclass bulirsch_stoer_dense_out[vect,num]:
        bulirsch_stoer_dense_out(num, num, num, num, bool)
        void initialize(vect&x0, num&t0, num&dt0)
        void calc_state(num&t, vect&x)
        vect&current_state()
        vect&previous_state()
        num current_time()
        num previous_time()
    # WARNING: boost's float128s use enough template magic that they can make
    # compilation melt down when handed the gigantic generated RHSs that come
    # out of sympy. One way to fix this would be to instead pass __float128s
    # into that function.

    #void do_step_dense(cKeplerRHS[quad]&rhs,
    #        bulirsch_stoer_dense_out[vectq,quad]&stepper)
    #void previous_state(bulirsch_stoer_dense_out[vectq,quad]&stepper,
    #        vectq&x)
    #void current_state(bulirsch_stoer_dense_out[vectq,quad]&stepper,
    #        vectq&x)

    void do_step_dense(cKeplerRHS[longdouble]&rhs,
            bulirsch_stoer_dense_out[vectl,longdouble]&stepper)
    void previous_state(bulirsch_stoer_dense_out[vectl,longdouble]&stepper,
            vectl&x)
    void current_state(bulirsch_stoer_dense_out[vectl,longdouble]&stepper,
            vectl&x)

    cdef cppclass bulirsch_stoer[vect,num]:
        bulirsch_stoer(num, num, num, num) except +
    void integrate_to(cKeplerRHS[longdouble] rhs,
            bulirsch_stoer[vectl,longdouble]&stepper,
            vectl&x, longdouble&t, longdouble&dt, longdouble t1) except +
    void integrate_to(cKeplerRHS[quad] rhs,
            bulirsch_stoer[vectq,quad]&stepper,
            vectq&x, quad&t, quad&dt, quad t1) except +
    void integrate_to(CRHS[quad] rhs,
            bulirsch_stoer[vectq,quad]&stepper,
            vectq&x, quad&t, quad&dt, quad t1) except +

cdef extern from "boost/numeric/odeint.hpp" namespace "boost::numeric::odeint":
    # This has got to go into C++ or the type inference is a nightmare
    pass



cdef vectq* array_to_vectq(np.ndarray[DTYPE_t, ndim=1] x) except NULL:
    cdef vectq* v = new vectq()
    for i in range(x.shape[0]):
        v.push_back(x[i])
    return v
cdef np.ndarray[DTYPE_t, ndim=1] vectq_to_array(vectq* v):
    cdef np.ndarray[DTYPE_t, ndim=1] x
    x = np.empty(v.size(), dtype=DTYPE)
    for i in range(v.size()):
        x[i] = quad_to_longdouble(v.at(i))
    return x

cdef vectl* array_to_vectl(np.ndarray[DTYPE_t, ndim=1] x) except NULL:
    cdef vectl* v = new vectl()
    for i in range(x.shape[0]):
        v.push_back(x[i])
    return v
cdef np.ndarray[DTYPE_t, ndim=1] vectl_to_array(vectl* v):
    cdef np.ndarray[DTYPE_t, ndim=1] x
    x = np.empty(v.size(), dtype=DTYPE)
    for i in range(v.size()):
        x[i] = v.at(i)
    return x

# FIXME: these suck.
@cython.boundscheck(False)
cdef quad_to_py(quad x):
    # this is strange-looking because assigning to a cdefed
    # array doesn't go through a python float, but reading
    # out from it goes through a python float rather than
    # an array scalar. But you can copy it and index the
    # non-cdefed array and get out an array scalar.
    cdef np.ndarray[DTYPE_t, ndim=1] v
    v = np.empty(1, dtype=DTYPE)
    v[0] = quad_to_longdouble(x)
    return v.copy()[0]
@cython.boundscheck(False)
cdef quad py_to_quad(x) except (<quad> FLT128_MAX):
    cdef np.ndarray[DTYPE_t, ndim=1] xa
    xa = np.array([x],dtype=DTYPE)
    return xa[<unsigned>0]
@cython.boundscheck(False)
cdef longdouble_to_py(DTYPE_t x):
    # this is strange-looking because assigning to a cdefed
    # array doesn't go through a python float, but reading
    # out from it goes through a python float rather than
    # an array scalar. But you can copy it and index the
    # non-cdefed array and get out an array scalar.
    cdef np.ndarray[DTYPE_t, ndim=1] v
    v = np.empty(1, dtype=DTYPE)
    v[0] = x
    return v.copy()[0]
@cython.boundscheck(False)
cdef longdouble py_to_longdouble(x):
    cdef np.ndarray[DTYPE_t, ndim=1] xa
    xa = np.array([x],dtype=DTYPE)
    return xa[<unsigned>0]

def shapiro_delay_l(x,gamma):
    cdef vectl *_x
    _x = array_to_vectl(np.asarray(x,dtype=DTYPE))
    r = shapiro_delay(_x[0],py_to_longdouble(gamma))
    del _x
    return r

cdef class ODEDelay:
    cdef longdouble _t_psr
    cdef longdouble _t_d
    cdef longdouble _t_bb
    cdef longdouble _atol, _rtol
    cdef vectl* _x
    cdef quad _t_psr_q
    cdef quad _t_d_q
    cdef quad _t_bb_q
    cdef quad _atol_q, _rtol_q
    cdef vectq* _x_q
    cdef int _n
    cdef int using_quad

    cdef int shapiro
    cdef int roemer

    cdef bulirsch_stoer_dense_out[vectl,longdouble]* _stepper
    cdef cKeplerRHS[longdouble]* _krhs
    cdef bulirsch_stoer_dense_out[vectq,quad]* _stepper_q
    cdef cKeplerRHS[quad]* _krhs_q
    cdef object pyrhs

    def __init__(self, rhs, initial_value, t_d, initial_dt=1e-6,
            atol=1e-10, rtol=1e-10,
            shapiro=True, roemer=True,
            use_quad=False):
        self.using_quad = use_quad
        self._n = len(initial_value)
        self.shapiro = shapiro
        self.roemer = roemer
        self.pyrhs = rhs
        if isinstance(rhs,KeplerRHS): # FIXME: causes crashes for some reason
            self._krhs = (<KeplerRHS>(self.pyrhs))._krhs_l
            self._krhs_q = (<KeplerRHS>(self.pyrhs))._krhs
        else:
            raise NotImplementedError
        self._x = array_to_vectl(np.asarray(initial_value, dtype=DTYPE))
        self._x_q = array_to_vectq(np.asarray(initial_value, dtype=DTYPE))

        self._t_d = py_to_longdouble(t_d)
        self._t_d_q = py_to_quad(t_d)
        self._atol = py_to_longdouble(atol)
        self._atol_q = py_to_quad(atol)
        self._rtol = py_to_longdouble(rtol)
        self._rtol_q = py_to_quad(rtol)
        self._stepper = new bulirsch_stoer_dense_out[vectl,longdouble](
                atol, rtol, 1, 1, True)
        self._stepper.initialize(self._x[0], self._t_d, initial_dt)
        self._stepper_q = new bulirsch_stoer_dense_out[vectq,quad](
                py_to_quad(atol), py_to_quad(rtol),
                py_to_quad(1), py_to_quad(1), True)
        self._stepper_q.initialize(self._x_q[0], self._t_d_q,
                                   py_to_quad(initial_dt))

    def __dealloc__(self):
        del self._x
        del self._stepper
        del self._x_q
        del self._stepper_q

    cdef longdouble delay(self, vectl&x):
        cdef longdouble d
        d = 0
        if self.roemer:
            d += x.at(2)/86400.
        if self.shapiro:
            d += shapiro_delay(x,self._krhs.gamma)
        if False and x.size()>21: # This should include only propagation delays
            d += x.at(21)
        return d
    cdef quad delay_q(self, vectq&x):
        cdef quad d
        d = 0
        if self.roemer:
            d += x.at(2)/86400.
        if self.shapiro:
            d += shapiro_delay(x,self._krhs_q.gamma)
        if False and x.size()>21: # This should include only propagation delays
            d += x.at(21)
        return d

    cpdef integrate_to(self, t_bb):
        # We're using uncorrected times
        cdef longdouble _t_bb
        cdef longdouble d
        cdef longdouble before_t_d, after_t_d, temp_t_d
        cdef longdouble before_t_bb, after_t_bb, temp_t_bb
        cdef longdouble w, oldw
        cdef longdouble thresh = self._rtol

        cdef quad _t_bb_q
        cdef quad d_q
        cdef quad before_t_d_q, after_t_d_q, temp_t_d_q
        cdef quad before_t_bb_q, after_t_bb_q, temp_t_bb_q
        cdef quad w_q, oldw_q
        cdef quad thresh_q = self._rtol_q

        if self.using_quad:
            _t_bb_q = py_to_quad(t_bb)
            while True:
                current_state(self._stepper_q[0], self._x_q[0])
                after_t_d_q = self._stepper_q.current_time()
                d_q = self.delay_q(self._x_q[0])
                if after_t_d_q+d_q>_t_bb_q:
                    break
                do_step_dense(self._krhs_q[0],self._stepper_q[0])
            # compute state by interpolation

            after_t_bb_q = after_t_d_q+d_q

            before_t_d_q = self._stepper_q.previous_time()
            previous_state(self._stepper_q[0], self._x_q[0])
            before_t_bb_q = before_t_d_q + self.delay_q(self._x_q[0])

            if before_t_bb_q>=_t_bb_q:
                raise ValueError("ODE integrator not started early enough for first data point")


            oldw_q = 10*(after_t_d_q-before_t_d_q)
            assert oldw_q>0
            while True:
                w_q = after_t_d_q-before_t_d_q
                assert w_q>0
                if w_q/oldw_q>0.75: # regula falsi is being slow
                    temp_t_d_q = (before_t_d_q+after_t_d_q)/2
                else: # use regula falsi - treat the function as linear
                    temp_t_d_q = (after_t_d_q*(_t_bb_q-before_t_bb_q)+
                               before_t_d_q*(after_t_bb_q-_t_bb_q))/(after_t_bb_q-before_t_bb_q)
                oldw_q = w_q

                self._stepper_q.calc_state(temp_t_d_q, self._x_q[0])
                temp_t_bb_q = temp_t_d_q+self.delay_q(self._x_q[0])
                if -thresh_q<temp_t_bb_q-_t_bb_q<thresh_q:
                    break

                if temp_t_bb_q>_t_bb_q:
                    after_t_d_q = temp_t_d_q
                    after_t_bb_q = temp_t_bb_q
                else:
                    before_t_d_q = temp_t_d_q
                    before_t_bb_q = temp_t_bb_q
                if after_t_d_q-before_t_d_q<thresh_q/100:
                    raise ValueError("""Convergence failure:
                        before_t_d = %s
                        after_t_d = %s
                        before_t_bb = %s
                        t_bb = %s
                        after_t_bb = %s
                        """ % (quad_to_py(before_t_d_q),
                               quad_to_py(after_t_d_q),
                               quad_to_py(before_t_bb_q),
                               quad_to_py(_t_bb_q),
                               quad_to_py(after_t_bb_q))
                               )
            self._t_d_q = temp_t_d_q
            self._t_bb_q = temp_t_bb_q
            self._t_psr_q = temp_t_d_q
            if self._x_q[0].size()>21: # if any time dilation
                self._t_psr_q -= self._x_q[0].at(21) # FIXME: is this sign right?
        else:
            _t_bb = py_to_longdouble(t_bb)
            while True:
                current_state(self._stepper[0], self._x[0])
                after_t_d = self._stepper.current_time()
                d = self.delay(self._x[0])
                if after_t_d+d>_t_bb:
                    break
                do_step_dense(self._krhs[0],self._stepper[0])
            # compute state by interpolation

            after_t_bb = after_t_d+d

            before_t_d = self._stepper.previous_time()
            previous_state(self._stepper[0], self._x[0])
            before_t_bb = before_t_d + self.delay(self._x[0])

            if before_t_bb>=_t_bb:
                raise ValueError("ODE integrator not started early enough for first data point")


            oldw = 10*(after_t_d-before_t_d)
            assert oldw>0
            while True:
                w = after_t_d-before_t_d
                assert w>0
                if w/oldw>0.75: # regula falsi is being slow
                    temp_t_d = (before_t_d+after_t_d)/2
                else: # use regula falsi - treat the function as linear
                    temp_t_d = (after_t_d*(_t_bb-before_t_bb)+
                               before_t_d*(after_t_bb-_t_bb))/(after_t_bb-before_t_bb)
                oldw = w

                self._stepper.calc_state(temp_t_d, self._x[0])
                temp_t_bb = temp_t_d+self.delay(self._x[0])
                if -thresh<temp_t_bb-_t_bb<thresh:
                    break

                if temp_t_bb>_t_bb:
                    after_t_d = temp_t_d
                    after_t_bb = temp_t_bb
                else:
                    before_t_d = temp_t_d
                    before_t_bb = temp_t_bb
                if after_t_d-before_t_d<thresh/100:
                    raise ValueError("Convergence failure")
            self._t_d = temp_t_d
            self._t_bb = temp_t_bb
            self._t_psr = temp_t_d
            if self._x[0].size()>21: # if any time dilation
                self._t_psr -= self._x[0].at(21) # FIXME: is this sign right?

    def __dealloc__(self):
        del self._x
        del self._stepper

    property x:
        def __get__(self):
            if self.using_quad:
                return vectq_to_array(self._x_q)
            else:
                return vectl_to_array(self._x)
    property t_d: # dynamical time - input to orbital motion
        def __get__(self):
            if self.using_quad:
                return quad_to_py(self._t_d_q)
            else:
                return longdouble_to_py(self._t_d)
    property t_psr: # pulsar proper time - time at psr center
        def __get__(self):
            if self.using_quad:
                return quad_to_py(self._t_psr_q)
            else:
                return longdouble_to_py(self._t_psr)
    property t_bb: # binary barycenter time - time pulse reaches binary barycenter
        def __get__(self):
            if self.using_quad:
                return quad_to_py(self._t_bb_q)
            else:
                return longdouble_to_py(self._t_bb)
    property n_evaluations:
        def __get__(self):
            return self.pyrhs.n_evaluations



cdef void class_callback(vector[quad]*x, vector[quad]*dxdt, quad t, void*arg) except *:
    rhs = (<RHS>arg)
    rhs._evaluate(x,dxdt,t)
cdef class RHS:
    cdef long long _n_evaluations
    cdef int _evaluate(self, vectq*x, vectq*dxdt, quad t) except -1:
        raise NotImplementedError
    cpdef evaluate(self, np.ndarray[DTYPE_t, ndim=1] x, t):
        cdef vectq* vx
        cdef vectq* vd
        cdef np.ndarray[DTYPE_t, ndim=1] r
        vx = array_to_vectq(x)
        vd = new vectq(x.shape[0],0)
        self._evaluate(vx, vd, py_to_quad(t))
        r = vectq_to_array(vd)
        del vx
        del vd
        return r
    property n_evaluations:
        def __get__(self):
            return self._n_evaluations

cdef class PyRHS(RHS):
    cdef object func
    def __init__(self, func):
        self.func = [func]
    cdef int _evaluate(self, vectq*x, vectq*dxdt, quad t) except -1:
        cdef np.ndarray[DTYPE_t, ndim=1] ax
        self._n_evaluations += 1
        pt = quad_to_py(t)
        ax = vectq_to_array(x)
        adxdt = self.func[0](ax, pt)
        for i in range(len(ax)):
            vectq_set(dxdt, i, adxdt[i])
        return 0
    cpdef evaluate(self, np.ndarray[DTYPE_t, ndim=1] x, t):
        return self.func[0](x, t)



cdef class KeplerRHS(RHS):
    cdef cKeplerRHS[quad]*_krhs
    cdef cKeplerRHS[longdouble]*_krhs_l
    cdef long long _evals
    def __init__(self, special=True, general=True, delta=0,
                 ppn_motion=False, gamma=1, beta=1, # default to GR
                 Gamma01=1, Gamma02=1, Gamma12=1,
                 Theta01=1, Theta02=1, Theta12=1,
                 Gamma011=1, Gamma012=1, Gamma022=1,
                 Gamma100=1, Gamma102=1, Gamma122=1,
                 Gamma200=1, Gamma201=1, Gamma211=1,
                 matrix_mode=0, c_scale=1):
        self._krhs=new cKeplerRHS[quad](special, general, self._evals,
            py_to_quad(delta), ppn_motion,
            py_to_quad(gamma), py_to_quad(beta),
            py_to_quad(Gamma01), py_to_quad(Gamma02), py_to_quad(Gamma12),
            py_to_quad(Theta01), py_to_quad(Theta02), py_to_quad(Theta12),
            py_to_quad(Gamma011), py_to_quad(Gamma012), py_to_quad(Gamma022),
            py_to_quad(Gamma100), py_to_quad(Gamma102), py_to_quad(Gamma122),
            py_to_quad(Gamma200), py_to_quad(Gamma201), py_to_quad(Gamma211),
            matrix_mode, py_to_quad(c_scale))
        self._krhs_l=new cKeplerRHS[longdouble](special, general, self._evals,
            py_to_longdouble(delta), ppn_motion,
            py_to_longdouble(gamma), py_to_longdouble(beta),
            py_to_longdouble(Gamma01), py_to_longdouble(Gamma02), py_to_longdouble(Gamma12),
            py_to_longdouble(Theta01), py_to_longdouble(Theta02), py_to_longdouble(Theta12),
            py_to_longdouble(Gamma011), py_to_longdouble(Gamma012), py_to_longdouble(Gamma022),
            py_to_longdouble(Gamma100), py_to_longdouble(Gamma102), py_to_longdouble(Gamma122),
            py_to_longdouble(Gamma200), py_to_longdouble(Gamma201), py_to_longdouble(Gamma211),
            matrix_mode, py_to_longdouble(c_scale))
    cdef int _evaluate(self, vectq*x, vectq*dxdt, quad t) except -1:
        self._krhs.evaluate(x, dxdt, t)
    def __dealloc__(self):
        del self._krhs
        del self._krhs_l
    property n_evaluations:
        def __get__(self):
            return self._evals
