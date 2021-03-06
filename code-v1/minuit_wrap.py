from __future__ import division, print_function
import sys
import inspect
import cPickle as pickle
import contextlib
import subprocess
import shlex
import functools32 as functools

import numpy as np
import scipy.optimize

import iminuit

@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)

class Fitter(object):
    def __init__(self, func, output=None, *args, **kwargs):
        if output is None:
            self.stdout = open('/dev/stdout', 'wt')
        else:
            self.stdout = output
        fargs = func.func_code.co_varnames
        if fargs[0] == 'self':
            fargs = fargs[1:]
        outer_self = self
        class Wrapfunc:
            def __init__(iself):
                class Thing: pass
                iself.func_code = Thing()
                iself.func_code.co_varnames = func.func_code.co_varnames
                iself.func_code.co_argcount = func.func_code.co_argcount
            def __call__(iself, *args):
                if self.printMode:
                    a = np.array(args)
                    cols = int(subprocess.check_output(['tput','cols']))
                    cols = max(cols-10,10)
                    with printoptions(linewidth=cols, precision=4):
                        if self.lastcall is None:
                            self.stdout.write(str(a))
                        else:
                            self.stdout.write(str(a-self.lastcall))
                        self.lastcall = a
                        self.stdout.flush()

                call_values = self._denormalize(fargs,args)
                r = func(*[v for (k,v) in call_values])
                if self.printMode:
                    self.stdout.write("\t%.6g" % r)
                    if self.best_values_fval is not None:
                        self.stdout.write("\t%.4g" % (r-self.best_values_fval))
                    self.stdout.write("\n")

                if self.best_values is None or r<self.best_values_fval:
                    self.best_values = call_values[:]
                    self.best_values_fval = r
                    if self.best_filename is not None:
                        with open(self.best_filename,"wb") as f:
                            pickle.dump(self.best_values, f)
                return r
        self.fargs = fargs
        self.wrapfunc2 = Wrapfunc()
        self._minuit = iminuit.Minuit(self.wrapfunc2,*args,pedantic=False,**kwargs)
        self.scale = dict((a,np.float128(1.)) for a in fargs)
        self.offset = dict((a,np.float128(0.)) for a in fargs)

        self.values = {}
        self.fixed = {}
        self.errors = {}
        for k in self._minuit.values:
            self.values[k] = np.float128(0)
            self.fixed[k] = False
            self.errors[k] = np.float128(1)
        self.best_values = None
        self.best_values_fval = None
        self.best_filename = None
        self.printMode = None
        self.lastcall = None

    def _normalize(self, ks, vs):
        return [(k,(v-self.offset[k])/self.scale[k])
                for (k,v) in zip(ks,vs)]
    def _denormalize(self, ks, vs):
        return [(k,np.float128(v)*self.scale[k]+self.offset[k])
                for (k,v) in zip(ks,vs)]

    def _set_minuit(self):
        for k,v in self._normalize(*zip(*self.values.items())):
            self._minuit.values[k] = v
            #self._minuit.fixed[k] = self.fixed[k]
            self._minuit.errors[k] = self.errors[k]/self.scale[k]
    def _get_minuit(self):
        for k,v in self._denormalize(*zip(*self._minuit.values.items())):
            self.values[k] = v
            self.errors[k] = self._minuit.errors[k]*self.scale[k]
    def set_normalization(self):
        for k in self.values:
            s = self.errors[k]
            if s==0:
                s = 1
            self.scale[k] = s
            self.offset[k] = self.values[k]

    def __getattr__(self, attrname):
        return getattr(self._minuit, attrname)

    #@property
    #def printMode(self):
    #    return self._minuit.printMode
    #@printMode.setter
    #def printMode(self, v):
    #    self._minuit.printMode = v
    @property
    def eps(self):
        return self._minuit.eps
    @eps.setter
    def eps(self, v):
        self._minuit.eps = v
    @property
    def tol(self):
        return self._minuit.tol
    @tol.setter
    def tol(self, v):
        self._minuit.tol = v
    @property
    def maxcalls(self):
        return self._minuit.maxcalls
    @maxcalls.setter
    def maxcalls(self, v):
        self._minuit.maxcalls = v
    @property
    def up(self):
        return self._minuit.up
    @up.setter
    def up(self, v):
        self._minuit.up = v
    @property
    def strategy(self):
        return self._minuit.strategy
    @strategy.setter
    def strategy(self, v):
        self._minuit.strategy = v

    def migrad(self, precision=None):
        self._set_minuit()
        r = self._minuit.migrad(precision=None)
        self._get_minuit()
        return r

    def simplex(self):
        self._set_minuit()
        r = self._minuit.simplex()
        self._get_minuit()
        return r

    def hesse(self):
        self._set_minuit()
        r = self._minuit.hesse()
        self._get_minuit()
        return r

    def matrix(self, correlation=False):
        M = self._minuit.matrix(correlation)
        if not correlation:
            rM = []
            for (i,p) in enumerate(self._minuit.parameters):
                rM.append([])
                for (j,q) in enumerate(self._minuit.parameters):
                    rM[-1].append(M[i][j]*self.scale[p]*self.scale[q])
            M = rM
        return M

    def scipy_minimize(self, method, **kwargs):
        def f(x):
            return f_t(tuple(x))
        @functools.lru_cache()
        def f_t(x):
            return self.wrapfunc2(*x)

        if self.best_values is None:
            x0 = np.zeros(len(self.fargs))
        else:
            bv = self._normalize(*zip(*self.best_values))
            # These values are normalized
            x0 = np.array([v for (k,v) in bv])
        if method=="basinhopping":
            scipy.optimize.basinhopping(f, x0, **kwargs)
        else:
            # FIXME: do something sensible with the result
            r = scipy.optimize.minimize(f, x0, method=method, **kwargs)
            # The optimizer result object contains best values, but so
            # does this object; we automatically keep best values ever seen

    def nlopt_minimize(self, optimizer=None, ftol_abs=1e-3, x0=None):
        import nlopt
        if optimizer is None:
            optimizer = nlopt.LN_BOBYQA
        opt = nlopt.opt(optimizer, len(self.fargs))
        def f(x, grad):
            return float(self.wrapfunc2(*x))
        opt.set_min_objective(f)
        opt.set_ftol_abs(1e-3)
        if x0 is None:
            if self.best_values is None:
                x0 = np.zeros(len(self.fargs))
            else:
                bv = self._normalize(*zip(*self.best_values))
                # These values are normalized
                x0 = np.array([v for (k,v) in bv])
        #x = opt.optimize(list(x0.astype(float)))
        x = opt.optimize(x0.astype(float))

    def gp_minimize(self, size=10, **kwargs):
        import skopt
        def f(x):
            return float(self.wrapfunc2(*x))
        if self.best_values is None:
            x0 = np.zeros(len(self.fargs))
        else:
            bv = self._normalize(*zip(*self.best_values))
            # These values are normalized
            x0 = np.array([v for (k,v) in bv])
        dimensions = [(-float(size),float(size)) for i in range(len(x0))]
        skopt.gp_minimize(f,dimensions,x0=x0,**kwargs)

    def basinhopping_minimize(self):
        def wrapped_bobyqa(fun, x0, args, kwargs, **options):
            self.nlopt_minimize(x0=x0)
            bv = self._normalize(*zip(*self.best_values))
            # These values are normalized
            x = np.array([v for (k,v) in bv])
            return scipy.optimize.OptimizeResult(
                x=x,
                success=True,
                status=0,
                message="",
                fun=self.best_values_fval,
                jac=None,
                hess=None,
                hess_inv=None,
                nfev=1,
                njev=0,
                nhev=0,
                nit=1,
                maxcv=0)
        
        pass
