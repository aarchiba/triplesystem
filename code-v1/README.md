# Code version 1

This code may never be imported to the project as-is. But at the least it is necessary to document how to use it, which largely means describing relevant notebooks.

## Generating the equations of motion

The notebook `n-ppn` contains a discussion of the problem and uses sympy to construct the Lagrangian and take appropriate derivatives of it to extract the equations of motion. These require some special handling to keep the equations even marginally manageable: I introduce symbols for the distances between bodies and squared velocities, for example. And of course I work in units where c is 1; G does not appear because rather than using the masses to parameterize the Lagrangian I use coefficients that appear in the Nordtvedt paper, that only equal products of the masses in GR. This notebook also contains some terms for tidal effects on the inner white dwarf.

The notebook `k-lagrangian-rhs` is a partial cleanup of the above, with the intention of generating an RHS that supports the computation of derivatives with respect to initial conditions and parameters. It is as yet incomplete, partly because generating workable C++ code from these very complicated expressions is nontrivial, and partly because it is necessary to choose a sensible, small set of parameters.

## Importing a new data set

To use a new collection of TOAs, it is necessary to assign them pulse numbers. One must then find a starting parameter set, then run an initial minimization with MINUIT. Getting an actually good best-fit set of values requires iterating between MINUIT minimization and MCMC runs; MINUIT is much faster at approaching the minimum but when it gets hung up on round-off error MCMC will often find a better solution. Be warned also that it can take a long time for MCMC to "notice" the SEP-violating solution.

