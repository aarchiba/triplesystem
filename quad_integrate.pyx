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
ctypedef vector[quad] vectq
ctypedef long double longdouble
ctypedef vector[longdouble] vectl


cdef extern from "extra.hpp":
    enum quad128values:
        FLT128_MAX
    void vectq_set(vectq*v, size_t i, quad e)
    void vectl_set(vectl*v, size_t i, longdouble e)
    ctypedef void (*callback_function)(
            vectq*x, vectq*dxdt, quad t, void* arg) except *

    cdef cppclass CRHS[num]:
        CRHS(void (*cb)(vector[num]*,vector[num]*,num,void*), void*arg) except +
        void evaluate(vector[num] &x, vector[num] &dxdt, num t) except +
    cdef cppclass cKeplerRHS[num]:
        cKeplerRHS(int special, int general, long long&evals) except +
        void evaluate(vector[num] *x, vector[num] *dxdt, num t) except +
        long long n_evaluations()

    #cdef cppclass bulirsch_stoer_dense_out[vect,num]:
    #    bulirsch_stoer_dense_out(num, num, num, num, bool)
    #    void initialize(vect&x0, num&t0, num&dt0)
    #void integrate_to_with_delay(CRHS[quad]&rhs, 
    #        bulirsch_stoer_dense_out[vectq,quad]&stepper,
    #        vectq&x, quad t)
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
        x[i] = v.at(i)
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
    cdef int use_quad

    cdef vectq* _x
    cdef quad _t
    cdef quad _dt
    cdef CRHS[quad]* _crhs
    cdef cKeplerRHS[quad]* _krhs
    cdef bulirsch_stoer[vectq,quad]* _stepper
    cdef quad delta

    cdef vectl* _x_l
    cdef longdouble _t_l
    cdef longdouble _dt_l
    cdef CRHS[longdouble]* _crhs_l
    cdef cKeplerRHS[longdouble]* _krhs_l
    cdef bulirsch_stoer[vectl,longdouble]* _stepper_l
    cdef longdouble delta_l

    cdef unsigned int _n
    cdef unsigned int _n_vec
    cdef object pyrhs
    cdef int symmetric

    def __init__(self, rhs, initial_value, t, initial_dt=0,
            atol=1e-10, rtol=1e-10, vectors=[], delta=1e-6,
            symmetric=False, use_quad=True):
        cdef unsigned int i, j

        self.use_quad = use_quad

        self._n = len(initial_value)
        self._n_vec = len(vectors)
        self.delta = delta
        self.delta_l = delta

        if not isinstance(rhs,RHS):
            raise ValueError("rhs must be an instance of the RHS class; try PyRHS")
        self.pyrhs = rhs # prevent freeing
        if isinstance(rhs,KeplerRHS):
            self._krhs = (<KeplerRHS>(self.pyrhs))._krhs
            self._krhs_l = (<KeplerRHS>(self.pyrhs))._krhs_l
        else:
            self._crhs = new CRHS[quad](&class_callback, <void*>rhs)
        

        self.symmetric = symmetric
        if self._n_vec>0:
            if self.symmetric:
                initial_value = np.concatenate(
                        [initial_value]*(2*self._n_vec+1))
            else:
                initial_value = np.concatenate(
                        [initial_value]*(self._n_vec+1))
        if self.use_quad:
            self._x = array_to_vectq(
                    np.asarray(initial_value, dtype=DTYPE))
        else:
            self._x_l = array_to_vectl(
                    np.asarray(initial_value, dtype=DTYPE))
        if self._n_vec>0:
            if self.symmetric:
                for i in range(self._n_vec):
                    for j in range(self._n):
                        if self.use_quad:
                            vectq_set(self._x, (2*i+1)*self._n+j,
                                    self._x.at((2*i+1)*self._n+j)
                                        -self.delta*py_to_quad(vectors[i][j]))
                            vectq_set(self._x, (2*i+2)*self._n+j,
                                    self._x.at((2*i+2)*self._n+j)
                                        +self.delta*py_to_quad(vectors[i][j]))
                        else:
                            vectl_set(self._x_l, (2*i+1)*self._n+j,
                                    self._x_l.at((2*i+1)*self._n+j)
                                        -self.delta_l*vectors[i][j])
                            vectl_set(self._x_l, (2*i+2)*self._n+j,
                                    self._x_l.at((2*i+2)*self._n+j)
                                        +self.delta_l*vectors[i][j])
            else:
                for i in range(self._n_vec):
                    for j in range(self._n):
                        if self.use_quad:
                            vectq_set(self._x, (i+1)*self._n+j,
                                      self._x.at((i+1)*self._n+j)
                                        +self.delta*py_to_quad(vectors[i][j]))
                        else:
                            vectl_set(self._x_l, (i+1)*self._n+j,
                                      self._x_l.at((i+1)*self._n+j)
                                        +self.delta_l*vectors[i][j])
        if self.use_quad:
            self._t = py_to_quad(t)
            self._dt = py_to_quad(initial_dt)
            self._stepper = new bulirsch_stoer[vectq,quad](
                    py_to_quad(atol), py_to_quad(rtol),
                    py_to_quad(1), py_to_quad(1))
        else:
            self._t_l = t
            self._dt_l = initial_dt
            self._stepper_l = new bulirsch_stoer[vectl,longdouble](
                    atol, rtol, 1, 1)
        
    cpdef integrate_to(self, t):
        cdef quad _t
        cdef longdouble _t_l
        if self.use_quad:
            _t = py_to_quad(t)

            if _t==self._t:
                return

            if self._dt==0:
                self._dt = 1e-4*(_t-self._t)

            if self._krhs!=NULL:
                integrate_to(self._krhs[0], 
                        self._stepper[0],
                        self._x[0],
                        self._t,
                        self._dt,
                        _t)
            else:
                integrate_to(self._crhs[0], 
                        self._stepper[0],
                        self._x[0],
                        self._t,
                        self._dt,
                        _t)
        else:
            _t_l = t
            if _t_l==self._t_l:
                return

            if self._dt_l==0:
                self._dt_l = 1e-4*(_t_l-self._t_l)

            if self._krhs_l!=NULL:
                integrate_to(self._krhs_l[0], 
                        self._stepper_l[0],
                        self._x_l[0],
                        self._t_l,
                        self._dt_l,
                        _t_l)
            else:
                raise NotImplementedError

    def __dealloc__(self):
        del self._x
        del self._crhs
        del self._stepper
        del self._stepper_l

    property x:
        def __get__(self):
            # convert to numpy array
            if self.use_quad:
                return vectq_to_array(self._x)[:self._n]
            else:
                return vectl_to_array(self._x_l)[:self._n]
    property dx:
        def __get__(self):
            cdef np.ndarray[DTYPE_t, ndim=2] result = np.empty((self._n,self._n_vec),dtype=DTYPE)
            cdef vectq *row
            cdef vectl *row_l
            cdef unsigned int i, j
            cdef quad deriv, l, r
            cdef longdouble deriv_l, l_l, r_l

            if self.use_quad:
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
            else:
                row_l = new vectl(self._n, 0)
                if self.symmetric:
                    for i in range(self._n_vec):
                        for j in range(self._n):
                            l_l = self._x_l.at((2*i+1)*self._n+j)
                            r_l = self._x_l.at((2*i+2)*self._n+j)
                            deriv_l = (r_l-l_l)/(2*self.delta_l)
                            vectl_set(row_l, j, deriv_l)
                        result[:,i] = vectl_to_array(row_l)
                else:
                    for i in range(self._n_vec):
                        for j in range(self._n):
                            l_l = self._x_l.at(j)
                            r_l = self._x_l.at((i+1)*self._n+j)
                            deriv_l = (r_l-l_l)/self.delta_l
                            vectl_set(row_l, j, deriv_l)
                        result[:,i] = vectl_to_array(row_l)
                del row_l

            return result
    property t:
        def __get__(self):
            if self.use_quad:
                return quad_to_py(self._t)
            else:
                return self._t_l
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

cdef class HarmonicRHS(RHS):
    cdef int _evaluate(self, vectq*x, vectq*dxdt, quad t) except -1:
        self._n_evaluations += 1
        vectq_set(dxdt, 0, x.at(1))
        vectq_set(dxdt, 1, -x.at(0))
        return 0


cdef class KeplerRHS(RHS):
    cdef cKeplerRHS[quad]*_krhs
    cdef cKeplerRHS[longdouble]*_krhs_l
    cdef long long _evals
    def __init__(self, special=True, general=True):
        self._krhs=new cKeplerRHS[quad](special, general, self._evals)
        self._krhs_l=new cKeplerRHS[longdouble](special, general, self._evals)
    cdef int _evaluate(self, vectq*x, vectq*dxdt, quad t) except -1:
        self._krhs.evaluate(x, dxdt, t)
    def __dealloc__(self):
        del self._krhs
        del self._krhs_l
    property n_evaluations:
        def __get__(self):
            return self._evals
