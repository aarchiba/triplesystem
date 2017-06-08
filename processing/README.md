Raw data processing
===================

The idea is to have a pipeline that can take raw data from all the telescopes and process it through into pulse time-of-arrival data. The general procedure follows the NANOGrav pipeline, though there is an additional step to update the ephemeris used by each file.

Raw data formats
----------------

We currently have raw data from three telescopes: the GBT, the Arecibo telescope, and the WSRT. The first two of these use nearly-identical backends, GUPPI and PUPPI, which produce a single PSRCHIVE-readable archive per observation (plus some calibration files). The WSRT produces no calibration files, but a raw WSRT observation is a large collection of archive files, one per subintegration per subband. We therefore add a nominally lossless pre-processing step to the WSRT data that combines all subintegrations and bands into a single archive file per observation.

* Combining the WSRT frequency bands requires re-aligning the archives using an ephemeris, so they shouldn't be combined until we've applied a short-term ephemeris.

* Looking at the per-Integration epoch attribute before and after alignment lets us estimate the ephemeris smearing in an observation; in some cases it was very large (e.g. 5% of a turn for one observation).

Once we have all our data in the form of a single aligned archive file per observation, we should be able to treat all our observations similarly:

* Apply a global per-telescope calibration file; this zaps always-bad channels
* Apply per-observation calibration files
* Use automatic RFI zapping based on the nanograv approach
* Tscrunch down to ~20-minute observations
* Fscrunch down to ~10-MHz channels (check SNR and scintillation)
* Make TOAs, discarding anything with SNR less than 10 (check value)

Note that this includes L-band and lower-frequency observations; for AO most observations do both frequencies within an hour, while for the GBT it may be days. The frequency scrunching for the lower-frequency observations is something we're going to have to look at, as is the template and JUMP value.


Things to watch out for
-----------------------

* WSRT telescope codes: i or j? one of these is baked into the raw data, and the version of tempo/tempo2 that interprets the ephemerides had better agree.
 * How do I find out what telescope code PSRCHIVE uses?
 * Just make tempo use both i and j for WSRT on dop263
* Change in tempo's BTX implementation - Marten van Kerkwijk's fixes made the BTX model in tempo work much better than it had previously. But has some of our data been taken with the old version? Will re-aligning these files work? Can we tell which version was used?
 * The key information is the per-Integration "epoch" value for each observation. This records the phase zero of the folded integration. The ephemeris originally used to fold the data is stored, but it need not be used to re-align the data to a new ephemeris.
 * Using psradd across frequencies requires a re-alignment with the stored ephemeris; this is asking for trouble if the stored ephemeris is somehow peculiar.
* On nimrod some files produce mysterious failures ("could not execute shell; insufficient resources") that are not present for the same files on dop263.

Code
----

The code exists as a couple of python modules driven by some ipython notebooks and standalone python programs. Entry points worth noting:

* `collect-uppi-data.ipynb`, `collect-puppi-data.ipynb`, `collect-wsrt-data.ipynb` - for importing new data
* `pac-preparation.ipynb` - for importing new cal scans
* `bulk-processing.ipynb` - for driving (re)processing of observations
* `make_toas.py` - script for processing large numbers of observations
* `summary.ipynb` - quick summary of the observations we have
* `template_match.py` - module and standalone script for Mueller matrix fitting
* `fit_segment.py` - Tool for producing short-term ephemerides
* `pipe.py` - module implementing most of the pipeline
