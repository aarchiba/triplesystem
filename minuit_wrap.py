
import inspect

import minuit


_fdef = """
def wrapfunc2(%s):
    return wrapfunc(%s)
"""

class MinuitWrap:
    def __init__(self, func, *args, **kwargs):
        fargs = inspect.getargspec(func).args
        def wrapfunc(*args):
            usvals = [v*self._scale[a]+self._offset[a] 
                        for (a,v) in zip(fargs,args)]
            return func(*usvals)
        s = ",".join(fargs)
        exec _fdef % (s,s)        
        self._minuit = Minuit(wrapfunc2,*args,**kwargs)
        self._scale = dict((a,1.) for a in fargs)
        self._offset = dict((a,0.) for a in fargs)

        svals = [(v-self.offset[a])/self.scale[a] 
                    for (a,v) in zip(fargs,args)]

        class Values(dict):
            def __getitem__(self2, k):
                return v-self.

