Raw data processing
===================

The idea is to have a pipeline that can take raw data from all the telescopes and process it through into pulse time-of-arrival data. The general procedure follows the NANOGrav pipeline, though there is an additional step to update the ephemeris used by each file.

Raw data formats
----------------

We currently have raw data from three telescopes: the GBT, the Arecibo telescope, and the WSRT. The first two of these use nearly-identical backends, GUPPI and PUPPI, which produce a single PSRCHIVE-readable archive per observation (plus some calibration files). The WSRT produces no calibration files, but a raw WSRT observation is a large collection of archive files, one per subintegration per subband. We therefore add a nominally lossless pre-processing step to the WSRT data that combines all subintegrations and bands into a single archive file per observation.

Once we have all our data in the form of a single archive file per observation, we should be able to treat all our observations similarly:

*



Things to watch out for
-----------------------

* WSRT telescope codes: i or j? one of these is baked into the raw data, and the version of tempo/tempo2 that interprets the ephemerides had better agree.
* Change in tempo's BTX implementation - Marten van Kerkwijk's fixes made the BTX model in tempo work much better than it had previously. But has some of our data been taken with the old version? Will re-aligning these files work? Can we tell which version was used?
