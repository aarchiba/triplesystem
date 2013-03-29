# distutils: language = c++
# distutils: include_dirs = include
# distutils: libraries = quadmath
# distutils: depends = extra.hpp
# distutils: extra_compile_args = -O -march=native

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
#ctypedef __float128 quad
ctypedef vector[quad] vectq

cdef extern from "extra.hpp":
    quad cpowq(quad,quad)
    quad cexpm1q(quad)
    quad clog1pq(quad)
    quad csqrtq(quad)
    enum quad128values:
        FLT128_MAX, M_Eq, M_PIq
    void vectq_set(vectq*v, size_t i, quad e)
    ctypedef void (*callback_function)(
            vectq*x, vectq*dxdt, quad t, void* arg) except *
    cdef cppclass CRHS:
        CRHS(callback_function cb, void*arg) except +
        void evaluate(vectq*x, vectq*dxdt, quad t) except *
        long long n_evaluations()
    void change_x(vectq&x)
    vector[vectq]* integrate_quad(CRHS&RHS, vectq&initial_state, vectq&times,
            quad atol, quad rtol, quad initial_dt)
    cdef cppclass bulirsch_stoer[vectq,quad]:
        bulirsch_stoer(quad, quad)
    cdef cppclass bulirsch_stoer_dense_out[vectq,quad]:
        bulirsch_stoer_dense_out(quad, quad, quad, quad, bool)
        void initialize(vectq&x0, quad&t0, quad&dt0)
    void integrate_to(CRHS&RHS, bulirsch_stoer[vectq,quad]&stepper,
            vectq&x, quad&t, quad&dt, quad t1)
    void integrate_to_with_delay(CRHS&RHS, 
            bulirsch_stoer_dense_out[vectq,quad]&stepper,
            vectq&x, quad t)

cdef extern from "boost/numeric/odeint.hpp" namespace "boost::numeric::odeint":
    # This has got to go into C++ or the type inference is a nightmare
    # Only works with doubles
    #cdef size_t integrate(CRHS, vectq&, quad, quad, quad) except 0
    #cdef cppclass runge_kutta4[State,Value]:
    #    pass
    #cdef size_t integrate_const(runge_kutta4[vectq,quad], CRHS, vectq&, quad, quad, quad) except 0
    #cdef cppclass runge_kutta_cash_karp54[State]:
    #    pass
    #ctypedef runge_kutta_cash_karp54[vectq] error_stepper_type 
    #ctypedef default_error_checker[quad] error_checker_type 
    #cdef make_controlled(quad, quad, error_stepper_type)
    #ctypedef controlled_runge_kutta[error_stepper_type,error_checker_type,initially_resizer] controlled_stepper_type
    #cdef size_t integrate_adaptive(controlled_stepper_type, CRHS, vectq&, quad, quad, quad) except 0
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
        x[i] = v.at(i)
    return x

# FIXME: these suck. 
@cython.boundscheck(False)
cdef quad_to_py(DTYPE_t x):
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
cdef quad py_to_quad(x) except (<quad> FLT128_MAX):
    cdef np.ndarray[DTYPE_t, ndim=1] xa
    xa = np.array([x],dtype=DTYPE)
    return xa[<unsigned>0]


cdef class ODE:
    cdef vectq* _x
    cdef vectq* _x0
    cdef quad _t
    cdef quad _dt
    cdef unsigned int _n
    cdef unsigned int _n_vec
    cdef CRHS* _crhs
    cdef bulirsch_stoer_dense_out[vectq,quad]* _stepper
    cdef object pyrhs
    cdef int symmetric
    cdef quad delta

    def __init__(self, rhs, initial_value, t, initial_dt=0,
            atol=1e-10, rtol=1e-10, vectors=[], delta=1e-6,
            symmetric=False):
        cdef unsigned int i, j
        if not isinstance(rhs,RHS):
            raise ValueError("rhs must be an instance of the RHS class; try PyRHS")
        self._crhs = new CRHS(class_callback, <void*>rhs)
        self.pyrhs = rhs # prevent freeing

        self._n = len(initial_value)
        self._n_vec = len(vectors)
        self.delta = delta

        self.symmetric = symmetric
        if self._n_vec>0:
            if self.symmetric:
                initial_value = np.concatenate(
                        [initial_value]*(2*self._n_vec+1))
            else:
                initial_value = np.concatenate(
                        [initial_value]*(self._n_vec+1))
        self._x = array_to_vectq(
                np.asarray(initial_value, dtype=DTYPE))
        self._x0 = array_to_vectq(
                np.asarray(initial_value, dtype=DTYPE))
        if self._n_vec>0:
            if self.symmetric:
                for i in range(self._n_vec):
                    for j in range(self._n):
                        vectq_set(self._x0, (2*i+1)*self._n+j,
                                self._x0.at((2*i+1)*self._n+j)
                                    -self.delta*py_to_quad(vectors[i][j]))
                        vectq_set(self._x0, (2*i+2)*self._n+j,
                                self._x0.at((2*i+2)*self._n+j)
                                    +self.delta*py_to_quad(vectors[i][j]))
            else:
                for i in range(self._n_vec):
                    for j in range(self._n):
                        vectq_set(self._x0, (i+1)*self._n+j,
                                  self._x0.at((i+1)*self._n+j)
                                    +self.delta*py_to_quad(vectors[i][j]))
        self._t = py_to_quad(t)
        self._dt = py_to_quad(initial_dt)
        self._stepper = new bulirsch_stoer_dense_out(
                py_to_quad(atol), py_to_quad(rtol),
                py_to_quad(1), py_to_quad(1),
                True)
        if self._dt==0:
            self._dt = 1e-4
        self._stepper.initialize(self._x0[0], self._t, self._dt)
    
    cpdef integrate_to(self, t):
        cdef quad _t
        _t = py_to_quad(t)
        integrate_to_with_delay(self._crhs[0], 
                self._stepper[0],
                self._x[0],
                _t)
        self._t = _t

    def __dealloc__(self):
        del self._x
        del self._x0
        del self._crhs
        del self._stepper

    property x:
        def __get__(self):
            # convert to numpy array
            return vectq_to_array(self._x)[:self._n]
    property dx:
        def __get__(self):
            cdef np.ndarray[DTYPE_t, ndim=2] result = np.empty((self._n,self._n_vec),dtype=DTYPE)
            cdef vectq *row
            cdef unsigned int i, j
            cdef quad deriv, l, r

            row = new vectq(self._n, 0)
            if self.symmetric:
                for i in range(self._n_vec):
                    for j in range(self._n):
                        l = self._x.at((2*i+1)*self._n+j)
                        r = self._x.at((2*i+2)*self._n+j)
                        deriv = (r-l)/(2*self.delta)
                        vectq_set(row, j, deriv)
                    result[:,i] = vectq_to_array(row)
            else:
                for i in range(self._n_vec):
                    for j in range(self._n):
                        l = self._x.at(j)
                        r = self._x.at((i+1)*self._n+j)
                        deriv = (r-l)/self.delta
                        vectq_set(row, j, deriv)
                    result[:,i] = vectq_to_array(row)
            del row

            return result
    property t:
        def __get__(self):
            # convert to numpy array
            return quad_to_py(self._t)
    property n_evaluations:
        def __get__(self):
            return self.pyrhs.n_evaluations

cpdef integrate_times(
        RHS rhs, initial_state,
        times, atol=1e-10, rtol=1e-10, 
        initial_dt = None):
    cdef vectq*i_s=NULL
    cdef vectq*ts=NULL
    cdef vector[vectq]*r=NULL
    cdef np.ndarray[DTYPE_t, ndim=2] ra
    cdef size_t i
    cdef CRHS *crhs=NULL

    times = np.asarray(times,dtype=DTYPE)
    initial_state = np.asarray(initial_state,dtype=DTYPE)

    if len(times.shape)!=1:
        raise ValueError("times must be one-dimensional but has shape %s" % (times.shape,))
    if len(initial_state.shape)!=1:
        raise ValueError("initial_state must be one-dimensional but has shape %s" % (initial_state.shape,))
    if initial_dt is None:
        if len(times)>1:
            initial_dt = times[1]-times[0]
        else:
            initial_dt = 0

    try:
        crhs = new CRHS(class_callback, <void*>rhs)
        i_s = array_to_vectq(initial_state)
        ts = array_to_vectq(times)
        r = integrate_quad(crhs[0], i_s[0], ts[0], 
                py_to_quad(atol), py_to_quad(rtol),
                py_to_quad(initial_dt))
        ra = np.empty((ts.size(), i_s.size()),dtype=DTYPE)
        for i in range(ts.size()):
            ra[i] = vectq_to_array(&r.at(i))
    finally:
        del i_s
        del ts
        del r
        del crhs
    return ra

cdef void class_callback(vectq*x, vectq*dxdt, quad t, void*arg) except *:
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

cdef class HarmonicRHS(RHS):
    cdef int _evaluate(self, vectq*x, vectq*dxdt, quad t) except -1:
        self._n_evaluations += 1
        vectq_set(dxdt, 0, x.at(1))
        vectq_set(dxdt, 1, -x.at(0))
        return 0



###########################################################################

# units are days, light-seconds, and solar masses
cdef quad G_mks = py_to_quad(6.67398e-11) # m**3 kg**(-1) s**(-2)
cdef quad c = py_to_quad(299792458) # m/s
cdef quad M_sun = py_to_quad(1.9891e30) # kg
cdef quad G = G_mks * c**(-3) * M_sun * 86400**2
cdef quad c3o2 = py_to_quad(3)/2

@cython.boundscheck(False)
cdef void kepler_inner(quad*x, quad*dxdt, quad t):
    cdef unsigned int n,i,j,k
    cdef quad m_i, m_j
    cdef quad x_i, y_i, z_i
    cdef quad x_j, y_j, z_j
    cdef quad r2_ij, cst

    for i in range(21):
        dxdt[i] = 0

    for i in range(3):
        for k in range(3):
            dxdt[7*i+k] = x[7*i+k+3]
        m_i = x[7*i+6]
        for j in range(i):
            m_j = x[7*j+6]
            
            r2_ij = 0
            for k in range(3):
                r2_ij += cpowq(x[7*j+k]-x[7*i+k],2)

            for k in range(3):
                cst = G*(x[7*j+k]-x[7*i+k])*cpowq(r2_ij,-c3o2)
                dxdt[7*i+k+3] +=  m_j*cst
                dxdt[7*j+k+3] += -m_i*cst

cdef class KeplerRHS(RHS):
    cdef int _evaluate(self, vectq*x, vectq*dxdt, quad t) except -1:
        cdef quad x_r[21]
        cdef quad dxdt_r[21]
        cdef unsigned int n,i,j,k
        cdef quad m_i, m_j
        cdef quad x_i, y_i, z_i
        cdef quad x_j, y_j, z_j
        cdef quad r2_ij, cst

        if x.size()>=21:
            for j in range(x.size()//21):
                for i in range(21):
                    x_r[i] = x.at(i+21*j)
                kepler_inner(x_r, dxdt_r, t)
                for i in range(21):
                    vectq_set(dxdt, i+21*j, dxdt_r[i])
        else:
            for i in range(x.size()):
                vectq_set(dxdt, i, 0)

            for i in range(x.size()//7):
                for k in range(3):
                    vectq_set(dxdt, 7*i+k, x.at(7*i+k+3))
                m_i = x.at(7*i+6)
                for j in range(i):
                    m_j = x.at(7*j+6)
                    
                    r2_ij = 0
                    for k in range(3):
                        r2_ij += cpowq(x.at(7*j+k)-x.at(7*i+k),2)

                    for k in range(3):
                        cst = G*(x.at(7*j+k)-x.at(7*i+k))*cpowq(r2_ij,-c3o2)
                        vectq_set(dxdt, 7*i+k+3,
                                dxdt.at(7*i+k+3)+m_j*cst)
                        vectq_set(dxdt, 7*j+k+3,
                                dxdt.at(7*j+k+3)-m_i*cst)
        return 0
cdef class KeplerRHSDoppler(RHS):
    cdef int _evaluate(self, vectq*x, vectq*dxdt, quad t) except -1:
        cdef quad x_r[21]
        cdef quad dxdt_r[21]
        cdef unsigned int i,j
        cdef quad vz

        for j in range(x.size()//21):
            for i in range(21):
                x_r[i] = x.at(i+21*j)
            kepler_inner(x_r, dxdt_r, t)
            vz = x_r[2]
            for i in range(21):
                dxdt_r[i]  /= (1+vz)
            for i in range(21):
                vectq_set(dxdt, i+21*j, dxdt_r[i])

        return 0
cdef class KeplerRHSRelativity(RHS):
    cdef int special, general
    cdef quad c2, G
    def __init__(self, special=True, general=True):
        self.special = special
        self.general = general
        self.c2 = 86400*86400
        self.G = 36779.59091405234
    cdef int _evaluate(self, vectq*x, vectq*dxdt, quad t) except -1:
        cdef quad x_r[21]
        cdef quad dxdt_r[21]
        cdef unsigned int i
        cdef quad slowing
        cdef quad temp, r, v2

        for i in range(21):
            x_r[i] = x.at(i)
        kepler_inner(x_r, dxdt_r, t)
        for i in range(21):
            vectq_set(dxdt, i, dxdt_r[i])
        slowing = 0
        if self.special:
            v2 = x_r[3]*x_r[3]+x_r[4]*x_r[4]+x_r[5]*x_r[5]
            slowing = cexpm1q(-0.5*clog1pq(-v2/self.c2))
        if self.general:
            r = (x_r[7]-x_r[0])*(x_r[7]-x_r[0])
            r += (x_r[8]-x_r[1])*(x_r[8]-x_r[1])
            r += (x_r[9]-x_r[2])*(x_r[9]-x_r[2])
            r = csqrtq(r)
            temp = cexpm1q(-0.5*clog1pq(-2*self.G*x_r[13]/(r*self.c2)))
            slowing = slowing + temp + slowing*temp

            r = (x_r[14]-x_r[0])*(x_r[14]-x_r[0])
            r += (x_r[15]-x_r[1])*(x_r[15]-x_r[1])
            r += (x_r[16]-x_r[2])*(x_r[16]-x_r[2])
            r = csqrtq(r)
            temp = cexpm1q(-0.5*clog1pq(-2*self.G*x_r[20]/(r*self.c2)))
            slowing = slowing + temp + slowing*temp

        vectq_set(dxdt,21,slowing)
        return 0
