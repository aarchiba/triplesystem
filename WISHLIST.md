# Wishlist items for the code

## Major

### Derivatives directly from the DE solver

A general-purpose ODE solver can be made to produce derivatives of the solution with respect to parameters or initial values if the RHS is supplemented with the corresponding derivatives of the original RHS; the higher-dimensional system can then be integrated normally. This would be a handy ability for the ODE solver here to have.

* Getting the derivatives of the RHS is a Simple Matter of Programming, since we already generate it using symbolic algebra.
* This removes issues of numerical roughness from the calculation of derivatives, for example for MINUIT optimization.
* Having ready derivatives would allow the use of techniques like Hamiltonian Monte Carlo to speed up convergence (but they're not particularly parallelizable).

## Minor

* Full covariance matrix for the errors. This is necessary to handle intrinsic pulsar jitter (which is correlated between all TOAs taken simultaneously), per-day DMs (which introduce the same error to each TOA they're applied to), and red noise. All but the last involve a sparse covariance matrix, but sparse Cholesky factorization, while available, is not in scipy and is kind of complicated. And red noise is not sparse.
* Linear fitting for all(?) non-orbit parameters. This may require splitting some of them up into an initial value and a correction that is handled linearly. It should reduce the dimensionality of the parameter space and accelerate the fitting substantially. More, it should allow introduction of many parameters (e.g. per-day DMs) into the general-purpose fitting.