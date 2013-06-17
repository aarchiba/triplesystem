
import inspect

import numpy as np

import minuit

_fdef = """
def wrapfunc2(%s):
    return wrapfunc(%s)
"""

class Fitter(object):
    def __init__(self, func, *args, **kwargs):
        fargs = inspect.getargspec(func).args
        if fargs[0] == 'self':
            fargs = fargs[1:]
        def wrapfunc(*args):
            r = func(**dict(self._denormalize(fargs,args)))
            if self.best_values is None or r<self.best_values_fval:
                self.best_values = dict(self._denormalize(fargs,args))
                self.best_values_fval = r
            return r
        s = ",".join(fargs)
        exec _fdef % (s,s) in locals()
        self._minuit = minuit.Minuit(wrapfunc2,*args,**kwargs)
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

    def _normalize(self, ks, vs):
        return [(k,(v-self.offset[k])/self.scale[k])
                for (k,v) in zip(ks,vs)]
    def _denormalize(self, ks, vs):
        return [(k,np.float128(v)*self.scale[k]+self.offset[k])
                for (k,v) in zip(ks,vs)]

    def _set_minuit(self):
        for k,v in self._normalize(*zip(*self.values.items())):
            self._minuit.values[k] = v
            self._minuit.fixed[k] = self.fixed[k]
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

    @property
    def printMode(self):
        return self._minuit.printMode
    @printMode.setter
    def printMode(self, v):
        self._minuit.printMode = v
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

    def migrad(self):
        self._set_minuit()
        r = self._minuit.migrad()
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

