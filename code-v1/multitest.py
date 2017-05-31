import multiprocessing

#import numpy as np

#import threebody

def f(x):
    return sum(range(1000000))
#F = threebody.Fitter()
#   return np.sum(F.residuals(F.best_parameters)**2/F.phase_errors)

if __name__=='__main__':
    p = multiprocessing.Pool(4)
    print p.map(f,range(1000000))

