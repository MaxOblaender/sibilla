"""sibilla — blind Zernike aberration recovery from 3D microscopy stacks.

Successor to Zoltan Cseresnyes' C# SIBILLA (see sibilla_zoltan). Determines
the optical aberration state of the imaging system as Zernike coefficients
directly from measured 3D data ("blind" — no separate wavefront measurement),
and builds the corresponding PSF for deconvolution.

Subpackages (flat for now; may be split into optics/fitting/data/pipeline
later if the project is shared externally).
"""

__version__ = "0.2.0"
