r"""
Horns.py
========

Terrain slope from a Digital Elevation Model (DEM) using HORN'S (1981)
finite-difference method over the complete 3x3 cell neighbourhood.

Purpose
-------
This module derives a *slope angle* raster (in degrees) from an *elevation*
raster (a DEM, in linear vertical units such as metres). The finite-difference
arithmetic is written out explicitly in NumPy rather than delegated to a
black-box GIS tool, so that every step of the numerical procedure is visible,
auditable, and reproducible by another researcher. No ArcPy, no GDAL slope
utility, and no third-party terrain library participates in the calculation:
the only operations applied to the elevation values are array slicing,
addition, subtraction, division, a square root and an arctangent.

It is the companion of ``slope_central_differences.py`` in this repository,
which implements the simpler second-order central difference. The two modules
share conventions, configuration format, validation rules and output metadata
style deliberately, so that the two slope products differ ONLY in the
derivative estimator and can therefore be compared directly.

Conceptual chain implemented here
---------------------------------
    Elevation  Z(x, y)                        [vertical linear units, e.g. m]
        |
        |  Horn's weighted finite differences over the full 3x3 neighbourhood
        v
    Partial derivatives  dZ/dx , dZ/dy        [dimensionless: rise / run]
        |
        |  Euclidean norm
        v
    Gradient magnitude  |grad Z|              [dimensionless]
        |
        |  arctangent
        v
    Slope angle  theta = arctan(|grad Z|)     [radians]
        |
        |  x 180/pi
        v
    Slope raster                              [DEGREES, 0 <= theta < 90]

The output is *slope* (equivalently *slope angle*). It is not "steepness
elevation"; it is not elevation of any kind, and it is not percent slope. Its
units are degrees of arc. See Section 8 for the terminology, which is used
consistently throughout this file.


1. Mathematical formulation
---------------------------

1.1 Why a finite-difference method is needed at all
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A DEM is a DISCRETE representation of a CONTINUOUS elevation surface

    Z = Z(x, y)

sampled on a regular grid of cells of size (dx, dy). Slope is defined from the
*gradient* of that surface, the vector of its first partial derivatives:

    grad Z = ( dZ/dx , dZ/dy )

The continuous partial derivatives are defined as limits,

    dZ/dx = lim_{h -> 0} [ Z(x + h, y) - Z(x, y) ] / h

but a DEM provides elevation only at the discrete sample points, never in the
limit: there is no elevation value between two cell centres, and the smallest
available h is one cell. The derivative therefore cannot be evaluated
analytically at a raster cell, and must be APPROXIMATED from the elevations of
the cells surrounding it. A rule that estimates a derivative from a finite set
of neighbouring samples separated by a finite spacing is, by definition, a
FINITE-DIFFERENCE method. Horn's method is one such rule; the central
difference is another.

1.2 The 3x3 neighbourhood
~~~~~~~~~~~~~~~~~~~~~~~~~
Horn's method reads the complete 3x3 block of cells centred on the focal cell.
The nine cells are labelled z1 to z9 in reading order::

                       y direction (north)
                              ^
                              |
            z1       z2       z3        <- row ABOVE the focal cell
             \       |       /             (array row i-1; north on a
              \      |      /               north-up raster)
            z4 ---- z5 ---- z6          <- the focal cell's own row
              /      |      \              (array row i)
             /       |       \
            z7       z8       z9        <- row BELOW the focal cell
                                           (array row i+1; south)

            <------ x direction (east) ------>
            column     column     column
              j-1        j         j+1

    z5             the FOCAL cell, the cell whose slope is being computed
    z2, z8         directly north and south of the focal cell
    z4, z6         directly west and east of the focal cell
    z1, z3, z7, z9 the four DIAGONAL neighbours

In terms of the NumPy array indices actually used in the implementation, with
``elevation[i, j]`` the focal cell::

   z1 = elevation[i-1, j-1]  z2 = elevation[i-1, j]  z3 = elevation[i-1, j+1]
   z4 = elevation[i,   j-1]  z5 = elevation[i,   j]  z6 = elevation[i,   j+1]
   z7 = elevation[i+1, j-1]  z8 = elevation[i+1, j]  z9 = elevation[i+1, j+1]

The first index is the ROW and increases downward through the array; on a
conventional north-up raster that is southward on the ground. The second index
is the COLUMN and increases rightward, which is eastward. Section 2 treats the
consequences of this for the sign of the y derivative.

Note that all eight cells surrounding the focal cell take part, and that the
focal elevation z5 itself does NOT appear in either derivative equation below.
This is not an oversight: a derivative is a rate of CHANGE, and Horn's rule
estimates it from the difference between the elevations on opposite sides of
the focal location. The focal cell sits exactly on the axis of each difference
and cancels out, just as it does in the ordinary central difference. The focal
cell's own validity is nevertheless required by this implementation, for the
reason given in Section 4.

1.3 Horn's partial derivatives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The derivative in the x (column, easting) direction is the weighted difference
between the three cells of the EAST column and the three cells of the WEST
column::

                (z3 + 2 z6 + z9) - (z1 + 2 z4 + z7)
    dZ/dx  =   -------------------------------------
                              8 dx

The derivative in the row direction is the weighted difference between the
three cells of the BOTTOM row and the three cells of the TOP row::

                (z7 + 2 z8 + z9) - (z1 + 2 z2 + z3)
    dZ/di  =   -------------------------------------
                              8 dy

Reading the x equation: each of the three east-column cells is compared with
the west-column cell in the same array row, giving three independent one-row
estimates of the east-west elevation difference. The middle row, which passes
through the focal cell, is given weight 2; the two outer rows, which reach the
focal cell only diagonally, are given weight 1 each. The weights sum to
2 + 1 + 1 = 4 on each side, and each pair of compared cells is separated by
2 dx, hence the denominator 4 * 2 dx = 8 dx. The same reading applies to the
row equation with the roles of rows and columns exchanged.

Equivalently, and this is the clearest way to see that the denominator is
correct, Horn's estimate is the WEIGHTED MEAN of three ordinary central
differences taken along three parallel lines::

    dZ/dx  =  [ 1 * (z3 - z1)/(2 dx)        <- central difference, row above
              + 2 * (z6 - z4)/(2 dx)        <- central difference, focal row
              + 1 * (z9 - z7)/(2 dx) ]      <- central difference, row below
              / (1 + 2 + 1)

    which expands to  [(z3 + 2 z6 + z9) - (z1 + 2 z4 + z7)] / (8 dx).

Each bracketed term is a slope estimate in its own right; Horn's rule averages
them with weights 1, 2, 1. The weighted average of three estimators of the
same quantity is itself an estimator of that quantity, with lower sensitivity
to the error in any single elevation sample. This is the sense in which Horn's
method gives a more spatially distributed estimate of the local gradient: it
draws on eight elevation observations instead of two per direction, and its
1-2-1 weighting acts as a small low-pass (smoothing) filter applied
TRANSVERSE to the direction of differentiation.

1.4 Horn's method compared with the simple central difference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The simple central difference uses only the two cells flanking the focal cell
in the direction of interest::

    dZ/dx  =  [ Z(i, j+1) - Z(i, j-1) ] / (2 dx)  =  (z6 - z4) / (2 dx)
    dZ/di  =  [ Z(i+1, j) - Z(i-1, j) ] / (2 dy)  =  (z8 - z2) / (2 dy)

It is the special case of the weighted average in Section 1.3 in which the two
off-centre rows are given weight 0. The two methods therefore differ in which
cells enter the estimate and with what weight, not in kind: both are
finite-difference estimators of the same continuous partial derivatives, and
both produce slope.

    Characteristic          Central difference        Horn's method
    ---------------------   -----------------------   -------------------------
    Neighbourhood           4 orthogonal neighbours   complete 3x3 window
    Cells read              z2, z4, z6, z8            all eight around z5
    Diagonal cells          not explicitly used       used, with weight 1
    Derivative estimation   two-point difference      weighted 3x3 difference
    Spatial weighting       direct neighbours only    orthogonal neighbours
                                                      weighted 2, diagonals 1
    Denominator             2 dx                      8 dx
    Output                  slope                     slope
    Method type             finite difference         finite difference

Consequences that can be stated objectively, without ranking the methods:

  (a) Response to elevation noise. Averaging three parallel differences
      reduces the variance contributed by uncorrelated vertical error. For
      independent, identically distributed cell errors of standard deviation
      s, the standard deviation of the estimated derivative is
      s / (dx * sqrt(2)) = 0.707 s / dx for the central difference and
      s * sqrt(1 + 4 + 1 + 1 + 4 + 1) / (8 dx) = s * sqrt(12) / (8 dx)
      = 0.433 s / dx for Horn's, about 39 % lower.

  (b) Response to real terrain detail. The same transverse averaging is
      indifferent to whether the variation it smooths is noise or genuine
      terrain. Horn's estimate is a property of a 3x3 window, so narrow
      features - a gully, a levee crest, a road cut, a terrace riser - are
      represented less sharply than by the central difference, and extreme
      values are attenuated.

  (c) Order of accuracy. On a smooth surface both are second-order accurate:
      the leading truncation error scales with the square of the cell size.
      Horn's estimate carries an additional term involving the transverse
      second derivative, because it averages across three rows, so the two
      agree exactly on a planar surface and differ where the surface is
      curved across the direction of differentiation.

  (d) Data requirement. The central difference needs 4 valid neighbours, Horn
      needs 8. Where NoData is extensive, Horn's method therefore yields a
      smaller valid area (Section 4).

Neither method is universally superior. Which is preferable depends on the
noise level of the DEM, the spatial scale of the terrain features of interest,
and the scale at which the slope raster will subsequently be used. Horn's
method is the estimator implemented by the slope tools of several mainstream
GIS packages, so it is also the conventional choice where comparability with
published terrain products matters.

1.5 Gradient magnitude
~~~~~~~~~~~~~~~~~~~~~~
    |grad Z| = sqrt( (dZ/dx)^2 + (dZ/dy)^2 )

Physically, grad Z is the vector pointing in the direction of STEEPEST ASCENT
of the elevation surface at that location, and its magnitude is the rate of
elevation change per unit horizontal distance travelled in that direction. It
is the maximum directional derivative over all azimuths: no direction of
travel across that cell gains elevation faster than |grad Z| metres per metre.
Because elevation and horizontal distance are expressed in the same linear
unit (Section 3.2), |grad Z| is DIMENSIONLESS - rise per unit run.

1.6 Slope angle
~~~~~~~~~~~~~~~
The gradient magnitude is the TANGENT of the angle between the terrain surface
and the horizontal plane. Inverting it gives the angle itself::

    theta          = arctan(|grad Z|)                [radians]
    theta_degrees  = arctan(|grad Z|) * 180 / pi     [degrees]

The output raster stores theta_degrees. Because arctan maps [0, inf) onto
[0, 90) degrees, the output is mathematically confined to

    0 <= theta < 90 degrees

for any finite gradient. Exactly 90 degrees is unattainable: it would require
an infinite gradient, and a single-valued surface Z(x, y) cannot represent a
vertical face or an overhang in any case. The implementation checks this range
as a quality-control assertion rather than enforcing it by clipping
(Section 7).

Percent slope is a DIFFERENT parameterisation of the same terrain::

    slope_percent = |grad Z| * 100

It is reported in the run log for contrast and is never written to the raster.
The two are not interchangeable: 100 % is 45 degrees, and percent slope is
unbounded above while degrees are bounded by 90.


2. Raster geometry and the orientation of the y derivative
----------------------------------------------------------

2.1 Array rows are not the y axis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A NumPy array indexes ``[row, column]`` with the row index increasing
DOWNWARD through the array. A conventional north-up GeoTIFF stores its first
row at the NORTHERN edge of the extent, which is why its affine transform
carries a NEGATIVE pixel-height term (``transform.e < 0``): stepping one row
down the array decreases the northing coordinate.

Horn's second equation, as written in Section 1.3 and as published, subtracts
the top row (z1, z2, z3) from the bottom row (z7, z8, z9). Its numerator is
therefore the elevation increase in the direction of INCREASING ROW INDEX,
which on a north-up raster is the increase toward the SOUTH. Written
faithfully, that quantity is

    dZ/di  = [(z7 + 2 z8 + z9) - (z1 + 2 z2 + z3)] / (8 dy)

and the geographic derivative with respect to NORTHING is

    dZ/dy  = orientation * dZ/di,     orientation = -1 for a north-up raster
                                                   +1 for a south-up raster

This module computes ``dZ/di`` exactly as published, then applies the
orientation factor read from the sign of ``transform.e``, so that the array
named ``dz_dy`` genuinely is the derivative with respect to northing. Only the
magnitude of ``transform.e`` is used as dy; its sign is used only for the
orientation.

2.2 Why the sign convention cannot affect the slope
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The y derivative enters the slope only through the gradient magnitude, where
it is SQUARED:

    |grad Z| = sqrt( (dZ/dx)^2 + (dZ/dy)^2 )   and   (-a)^2 = a^2

so negating dZ/dy leaves |grad Z|, and hence theta, exactly unchanged. Slope
is a magnitude and carries no directional information; a mistaken sign would
be invisible in a slope raster.

The sign is nevertheless applied correctly here for two reasons: the returned
``dz_dy`` array is documented as the derivative with respect to northing and
must therefore actually be that; and any DIRECTIONAL quantity derived later
from these components - aspect, hillshade, flow direction, curvature - depends
on the sign and would be reflected north-for-south if it were wrong.


3. Spatial reference requirements
---------------------------------

3.1 A projected CRS is required
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The denominators 8 dx and 8 dy must be genuine horizontal DISTANCES in the
same linear unit as the elevation numerator. This module therefore requires a
PROJECTED CRS and refuses a geographic one by default.

In a geographic CRS (EPSG:4326, for instance) the coordinates are ANGLES and
the cell size read from the transform is in DEGREES. Forming

    (metres of elevation) / (degrees of longitude)

is not a slope: the quotient is not dimensionless, and its arctangent is not
an angle of terrain inclination. Two further problems compound this. One
degree of longitude is not a fixed ground distance - it shrinks as
cos(latitude), from about 111.3 km at the equator to about 55.8 km at 60 N and
to zero at the poles - so a single scalar dx cannot describe a whole
geographic raster. And one degree of longitude and one degree of latitude
subtend different ground distances everywhere except the equator, so a cell
that is "square" in degrees is not square on the ground, and dx and dy would
be inconsistently scaled relative to one another.

The remedy is to reproject the DEM to a projected CRS with linear units (an
appropriate UTM zone, a national or state grid, or a local conformal or
equal-area projection) before running this module. The script stops with an
explanatory error rather than silently treating degrees as metres. The
``allow_geographic_crs`` escape hatch exists only for diagnostic use and warns
in the strongest terms that its output is not a slope.

3.2 Vertical units and the z-factor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The gradient is dimensionless only if the elevation unit and the horizontal
unit are the SAME unit. If they differ - most commonly a DEM in metres on a
grid in US survey feet - the elevation array must be converted first. The
z-factor multiplies elevation by a constant before differencing::

    metres elevation on a metre grid          -> z_factor = 1.0
    metres elevation on a US survey foot grid -> z_factor = 3.2808333
    feet elevation on a metre grid            -> z_factor = 0.3048

A GeoTIFF rarely encodes its vertical unit, so this cannot be detected
automatically; the script reports the CRS's horizontal linear unit so the
choice can be made and documented explicitly, and defaults to 1.0.

3.3 Map projection scale distortion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A projected CRS gives coordinates in linear units, but those units are
measured ON THE MAP PLANE, not on the ground. The point scale factor

    k = (distance on the map plane) / (distance on the ellipsoid)

means that the true ground separation of two cells whose nominal separation is
D is D / k. Dividing an elevation difference by the nominal cell size where
k != 1 therefore biases every derivative by exactly k. For UTM the effect is
at most about 0.1 %; for the polar stereographic grid of ArcticDEM
(EPSG:3413, standard parallel 70 N) it reaches about +4 % at 60 N, which is a
systematic bias that does not average out.

The correction is available here through ``apply_scale_factor_correction`` and
is off by default, matching the nominal-grid-unit convention of mainstream GIS
slope tools. The scale-factor field is computed by
``compute_scale_factor_field`` imported from ``slope_central_differences.py``,
so that both slope products in this repository use one identical,
single-source implementation of the correction and cannot drift apart. That
import is the ONLY code this module borrows; every step of Horn's method
itself is implemented here in full.


4. NoData handling
------------------
NoData cells carry no elevation observation. Their stored value is a SENTINEL
- a numerical placeholder such as -9999, -32768 or NaN that marks the cell as
unobserved and is not a measurement of the ground. (The term is used here in
its numerical-computing sense; it has no connection to the Sentinel
Earth-observation missions.)

Allowing a sentinel into the arithmetic would be catastrophic rather than
merely imprecise. A single -9999 m neighbour beside a 100 m focal cell on a
10 m grid contributes (100 - (-9999)) to a numerator that should hold a
difference of a few metres, producing a near-vertical slope where the terrain
may be flat - and it corrupts not one cell but every cell that uses the
sentinel as a neighbour. Treating NoData as zero elevation is the same error
in a less obvious disguise: it fabricates a cliff at the edge of every data
gap. Neither substitution, nor gap filling, nor a one-sided fallback formula
is applied here.

The rule applied is strict and conservative:

    A cell receives a valid slope value only if ALL NINE cells of its 3x3
    neighbourhood hold genuine, finite elevation observations. Otherwise the
    output cell is set to NoData.

The eight surrounding cells are required because all eight appear in Horn's
equations; an incomplete window has no defined Horn estimate. The focal cell
z5 is required too, even though it does not enter the arithmetic, because
writing a slope value at a location where the DEM records no ground would
assert terrain that was never observed, and because the companion
central-difference product applies the same rule, which keeps the two
comparable.

The practical consequence is that a NoData region in the DEM is dilated by
exactly one cell IN ALL EIGHT DIRECTIONS in the slope product - including
diagonally, which is where this differs from the central-difference product,
whose gaps grow only along the four cardinal directions. It is preferable for
scientific use to produce a smaller, fully defensible valid area than a
complete raster containing silently fabricated values.

Validity is determined from three independent criteria combined with logical
AND: the GDAL band mask (which honours the declared nodata value and any
internal or alpha mask), an explicit comparison against the declared nodata
value, and a finiteness test excluding NaN and +/- infinity. NaN in particular
cannot be caught by equality, since NaN != NaN.


5. Raster boundaries
--------------------
Horn's method requires a complete 3x3 window. The cells of the outermost row
and the outermost column on each side have no such window: they are missing
between three and five of their nine neighbours. They are therefore assigned
NoData, and Horn's slope is computed only where a full 3x3 neighbourhood
exists.

Edges are never wrapped to the opposite side of the raster: the cells at
opposite edges of a DEM are separated by the entire width of the study area
and are not neighbours in any physical sense. Nor is the edge padded by
replicating or extrapolating elevations, which would fabricate a
zero-gradient or invented-gradient ring indistinguishable from measurement in
the output.

The valid interior is therefore (rows - 2) x (cols - 2) cells - a negligible
loss for a large DEM, a material one for a small tile. Tiled processing should
read each tile with an overlapping buffer of at least one cell, compute on the
buffered tile, and discard the buffer afterwards, so that the interior of the
mosaic carries no internal seams of NoData.


6. Numerical precision
----------------------
All arithmetic is performed in float64 (IEEE 754 double precision, about 16
significant decimal digits), regardless of the storage type of the input band.
Three reasons:

  (a) Integer DEMs. An integer band would truncate the quotient of the finite
      difference toward zero, quantising slope into a few discrete values.
      The band is cast to float64 before any arithmetic.

  (b) Cancellation. Horn's numerator is a difference of two sums of four
      elevations each, of similar magnitude. On a DEM of large-magnitude
      elevations - ellipsoidal heights, or a grid in feet - the significant
      digits of the difference are the LAST few digits of the operands, so the
      subtraction cancels most of the precision it began with. float32 carries
      about 7 decimal digits; on a 2000 m elevation that leaves a resolution
      of roughly 10^-4 m, which is not comfortably below the sub-millimetre
      differences a high-resolution DEM can legitimately contain. float64
      leaves about 10^-13 m, which never limits the result.

  (c) NaN propagation. float64 represents NaN natively, so invalid cells can
      be neutralised by assignment rather than tracked separately, and any
      failure of the masking logic surfaces as a visible NaN rather than as a
      plausible-looking slope derived from a sentinel.

Storage is float32. A float32 degree value resolves to about 10^-5 degrees,
four to five orders of magnitude finer than the accuracy of any DEM-derived
slope estimate, and it halves the file size. The conversion happens once, at
the write step, after all computation is complete.

Memory. The implementation is fully vectorised: it holds the elevation array,
the validity mask, and a small number of interior-sized temporaries, with a
peak of roughly 8 to 10 float64 arrays the size of the raster. For a DEM too
large for RAM, the same arithmetic should be applied window by window using
``rasterio.windows`` with a one-cell overlap between windows, writing each
window to the output dataset as it is completed; the interior arithmetic is
identical, because Horn's estimate at a cell depends on nothing outside its
own 3x3 window. That extension is deliberately not built into this module, to
keep the numerical core readable.


7. Output and quality control
-----------------------------
The output is a single-band float32 GeoTIFF whose CRS, affine transform,
extent, width and height are inherited unchanged from the input, so that the
slope raster overlays the DEM cell for cell. Co-registration is what makes it
possible to stack the slope raster with other predictors and extract training
samples at identical locations; a slope raster with a different transform
would introduce a spatial offset into every subsequent analysis, and a
half-cell shift at 1 m resolution is enough to sample the wrong side of a
terrain break.

Provenance is recorded in GeoTIFF metadata tags: the derived quantity, the
method, the units, the neighbourhood, the boundary and NoData rules, the cell
size actually used, the z-factor, whether the scale-factor correction was
applied, and the source file. Every tag states a fact about how this specific
file was produced; no tag asserts equivalence with any GIS package's output.

Quality control reports cell counts and the minimum, maximum, mean, median and
standard deviation of the valid slope values, and audits the result for NaN,
+/- infinity and values outside the attainable range [0, 90). Values outside
that range are mathematically impossible for a correctly computed arctangent,
so their presence would indicate a defect in the implementation or a corrupted
input, not an unusual landscape. They are therefore reported loudly and are
never silently clipped: clipping would conceal the defect while leaving the
product wrong.


8. Terminology used in this module
----------------------------------
    elevation            Z(x, y), the height of the surface above the vertical
                         datum, in vertical linear units (metres). A scalar
                         field. This is the INPUT.
    elevation gradient   grad Z = (dZ/dx, dZ/dy), a VECTOR field: the pair of
                         first partial derivatives. Dimensionless.
    gradient magnitude   |grad Z|, a scalar: the length of that vector, the
                         maximum rate of rise per unit horizontal run.
                         Dimensionless. Equals tan(theta).
    slope angle          theta = arctan(|grad Z|), the inclination of the
                         surface from horizontal, in degrees. This is the
                         OUTPUT.
    percent slope        |grad Z| * 100, the same gradient expressed as a
                         percentage rather than an angle. Reported in the log,
                         never written to the raster.

"Steepness" is an informal word for the same idea; the scientifically precise
variable is SLOPE, and that is the name used for the product.


9. Place in the research workflow
---------------------------------
    LiDAR / UAV survey
        |
    high-resolution elevation raster (DEM)
        |
    HORN'S FINITE-DIFFERENCE SLOPE CALCULATION   <- this module
        |
    slope raster (degrees)
        |
    terrain predictor, or reference target, in a machine-learning model
        |
    salmonberry distribution classification / prediction

Horn's method is NOT machine learning. It is a deterministic numerical
terrain-analysis procedure: the same DEM, cell size and options always produce
exactly the same slope raster, there is nothing fitted or trained, and no
parameter of the method is learned from data. The machine-learning stage
occurs later and separately, when this slope raster and other variables are
used as predictors, or when it serves as the reference target that a model
built from satellite predictors is trained to reproduce. Keeping the two
stages distinct matters for the methodological description of the research: a
slope raster is an observation derived by arithmetic, while a model prediction
of slope is an estimate with an error distribution.

References
----------
Horn, B. K. P. (1981). Hill shading and the reflectance map. Proceedings of
the IEEE, 69(1), 14-47.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import rasterio
    from rasterio.crs import CRS
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "This script requires 'rasterio'. Install it with:\n"
        "    pip install rasterio\n"
        "or, in an ArcGIS Pro / conda environment:\n"
        "    conda install -c conda-forge rasterio\n"
        f"(import error: {exc})"
    )

# PyYAML is required only for --config. Guarded so that the positional-argument
# command line still works without it.
try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:  # pragma: no cover - environment guard
    _YAML_AVAILABLE = False

# The map-projection scale-factor correction (module docstring, Section 3.3) is
# imported from the companion module rather than reimplemented, so that both
# slope products in this repository share one single-source implementation of
# the correction and cannot drift apart. The import is guarded: the whole of
# Horn's method runs without it, and the correction is off by default.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_SCALE_FACTOR_IMPORT_ERROR: Optional[str] = None
try:
    from slope_central_differences import (  # noqa: E402
        compute_scale_factor_field,
        SCALE_FACTOR_WARN_THRESHOLD,
    )
    _SCALE_FACTOR_AVAILABLE = True
except Exception as _exc:  # pragma: no cover - environment guard
    _SCALE_FACTOR_AVAILABLE = False
    _SCALE_FACTOR_IMPORT_ERROR = str(_exc)
    SCALE_FACTOR_WARN_THRESHOLD = 1e-3


# --------------------------------------------------------------------------
# Module-level constants
# --------------------------------------------------------------------------

#: NoData value written to the output slope raster. Slope in degrees is
#: mathematically confined to [0, 90), so any negative sentinel is unambiguous
#: and can never collide with a legitimate slope value.
DEFAULT_OUTPUT_NODATA: float = -9999.0

#: Storage type of the output raster. Computation is in float64 (module
#: docstring, Section 6); float32 storage resolves degrees far below the
#: accuracy of any DEM-derived slope estimate.
OUTPUT_DTYPE: str = "float32"

#: Minimum raster dimension. Horn's method needs one cell on every side of the
#: focal cell, so a 3x3 grid is the smallest raster with any evaluable cell
#: (exactly one, at its centre).
MIN_RASTER_DIMENSION: int = 3

#: Number of cells in the Horn neighbourhood, all of which must be valid
#: before a slope value is written (module docstring, Section 4).
HORN_WINDOW_CELLS: int = 9

#: Upper bound of the attainable slope range in degrees. The arctangent of a
#: finite gradient is strictly less than this; it is used only as a
#: quality-control assertion, never to clip values (Section 7).
MAX_ATTAINABLE_SLOPE_DEGREES: float = 90.0

#: Plausibility bounds for elevation in metres, used ONLY to emit a soft
#: warning that the band may not be an elevation surface. Lowest exposed land
#: is about -430 m (Dead Sea shore); highest is 8849 m (Everest). Bathymetric
#: or non-metre DEMs legitimately fall outside this range, hence a warning and
#: never an error.
PLAUSIBLE_ELEVATION_RANGE_M: Tuple[float, float] = (-500.0, 9000.0)

LOGGER = logging.getLogger("Horns")


# --------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RasterGrid:
    """
    A validated elevation grid and the spatial parameters Horn's method needs.

    Attributes
    ----------
    elevation : numpy.ndarray
        2-D float64 array of elevation values, shape (rows, cols). Cells that
        are NoData, NaN or infinite have been replaced by ``numpy.nan`` so
        that no sentinel magnitude can leak into the arithmetic.
    valid_mask : numpy.ndarray
        2-D boolean array, True where ``elevation`` holds a genuine finite
        elevation observation. This is the authoritative record of validity;
        ``elevation`` alone cannot be trusted, because NaN comparisons are
        always False.
    cell_size_x : float
        Horizontal cell size dx along the x (column / easting) axis, in the
        CRS's linear units. Strictly positive.
    cell_size_y : float
        Horizontal cell size dy along the y (row / northing) axis, in the
        CRS's linear units. Strictly positive, and independent of
        ``cell_size_x`` so that non-square cells are handled correctly.
    y_axis_orientation : int
        +1 or -1. The factor converting the row-index (downward-positive)
        derivative dZ/di into the geographic (northward-positive) derivative
        dZ/dy. It is -1 for the usual north-up raster (module docstring,
        Section 2.1).
    transform : affine.Affine
        The raster's affine geotransform, carried through unmodified to the
        output so that the slope raster is co-registered with the input.
    crs : rasterio.crs.CRS
        The raster's coordinate reference system, carried through unmodified.
    linear_unit : str
        Human-readable name of the CRS's horizontal linear unit, used in
        reporting and in the output metadata.
    crs_type : str
        ``"projected"``, ``"geographic"`` or ``"unknown"``, reported in the
        log so that the basis of the cell size is never in doubt.
    nodata_value : float or None
        The NoData value declared by the input raster, reported in the log and
        recorded in the output metadata as provenance.
    profile : dict
        The input raster's rasterio profile, used as the template for the
        output profile.
    scale_factor_x, scale_factor_y : float or numpy.ndarray
        Point scale factor k of the map projection along each axis (module
        docstring, Section 3.3). Either a scalar, where k is effectively
        uniform over the raster, or a 2-D float64 field. Dividing the nominal
        cell size by k is algebraically identical to multiplying the
        derivative by k, which is the form used in the computation. Both are
        exactly 1.0 when the correction is disabled, leaving the arithmetic
        unchanged.
    scale_factor_applied : bool
        Whether the correction was actually applied, recorded in the output
        metadata so that a product's provenance states unambiguously whether
        its slopes are in nominal grid units or corrected to ground distance.
    scale_factor_summary : str
        Human-readable description of the correction, for logs and metadata.
    """

    elevation: np.ndarray
    valid_mask: np.ndarray
    cell_size_x: float
    cell_size_y: float
    y_axis_orientation: int
    transform: object
    crs: "CRS"
    linear_unit: str
    crs_type: str
    nodata_value: Optional[float]
    profile: dict
    scale_factor_x: object = 1.0
    scale_factor_y: object = 1.0
    scale_factor_applied: bool = False
    scale_factor_summary: str = "not applied (nominal grid units)"


@dataclass(frozen=True)
class SlopeResult:
    """
    The computed slope raster and the intermediate gradient fields.

    The derivative components are retained so that a caller can inspect or
    export them without recomputing, and so that a directional product
    (aspect, hillshade) can be derived later from exactly the same numbers
    that produced the slope.

    Attributes
    ----------
    slope_degrees : numpy.ndarray
        2-D float64 array of slope angle in degrees, range [0, 90).
        ``numpy.nan`` marks cells that could not be evaluated (the boundary
        ring, or an incomplete 3x3 Horn neighbourhood).
    dz_dx : numpy.ndarray
        2-D float64 array of the partial derivative of elevation with respect
        to easting. Dimensionless. NaN where not evaluable.
    dz_dy : numpy.ndarray
        2-D float64 array of the partial derivative of elevation with respect
        to northing, with the geographic sign convention applied (module
        docstring, Section 2.1). Dimensionless. NaN where not evaluable.
    gradient_magnitude : numpy.ndarray
        2-D float64 array of |grad Z|, dimensionless, equal to tan(theta).
        NaN where not evaluable.
    output_valid_mask : numpy.ndarray
        2-D boolean array, True exactly where ``slope_degrees`` holds a
        finite, defensible slope value.
    """

    slope_degrees: np.ndarray
    dz_dx: np.ndarray
    dz_dy: np.ndarray
    gradient_magnitude: np.ndarray
    output_valid_mask: np.ndarray


# --------------------------------------------------------------------------
# Scalar reference implementation (documentation and self-test)
# --------------------------------------------------------------------------


def horn_slope_from_window(
    z1: float, z2: float, z3: float,
    z4: float, z5: float, z6: float,
    z7: float, z8: float, z9: float,
    cell_size_x: float,
    cell_size_y: float,
    y_axis_orientation: int = -1,
    scale_factor_x: float = 1.0,
    scale_factor_y: float = 1.0,
) -> Tuple[float, float, float, float]:
    """
    Scalar reference implementation of Horn's slope for a single 3x3 window.

    This function is the single-cell statement of exactly the same arithmetic
    that :func:`calculate_horn_slope` applies to the whole array with
    vectorised NumPy slicing. It exists so that the method can be read,
    reasoned about and unit-tested one cell at a time, and so that the
    vectorised implementation can be verified against an unambiguous
    reference. It is not used in the production path over the raster, which is
    vectorised for performance.

    Parameters
    ----------
    z1, z2, z3 : float
        Elevations of the row ABOVE the focal cell, west to east:
        ``Z(i-1, j-1)``, ``Z(i-1, j)``, ``Z(i-1, j+1)``. On a north-up raster
        this is the northern row; z1 and z3 are diagonal neighbours and z2 is
        the direct north neighbour.
    z4, z5, z6 : float
        Elevations of the focal row, west to east: ``Z(i, j-1)``, ``Z(i, j)``,
        ``Z(i, j+1)``. z5 is the FOCAL cell, z4 the west and z6 the east
        neighbour.
    z7, z8, z9 : float
        Elevations of the row BELOW the focal cell, west to east:
        ``Z(i+1, j-1)``, ``Z(i+1, j)``, ``Z(i+1, j+1)``. On a north-up raster
        this is the southern row; z7 and z9 are diagonal neighbours and z8 is
        the direct south neighbour.
    cell_size_x : float
        dx, the cell size along the column axis, in the same linear unit as
        the elevations.
    cell_size_y : float
        dy, the cell size along the row axis, in the same linear unit as the
        elevations.
    y_axis_orientation : int, optional
        -1 for a north-up raster (default), +1 for a south-up raster. Converts
        the row-index derivative into the geographic derivative. Affects the
        sign of the returned ``dz_dy`` only, never the returned slope (module
        docstring, Section 2.2).
    scale_factor_x, scale_factor_y : float, optional
        Point scale factor k of the map projection along each axis, converting
        the nominal map-plane cell size into true ground distance (module
        docstring, Section 3.3). Default 1.0, i.e. no correction, reproducing
        the nominal-grid-unit convention of GIS slope tools.

    Returns
    -------
    tuple of (float, float, float, float)
        ``(dz_dx, dz_dy, gradient_magnitude, slope_degrees)``:

        - ``dz_dx``: partial derivative with respect to easting, dimensionless
        - ``dz_dy``: partial derivative with respect to northing, dimensionless
        - ``gradient_magnitude``: |grad Z|, dimensionless, equals tan(theta)
        - ``slope_degrees``: slope angle in degrees, in [0, 90)

    Method
    ------
    Horn's (1981) weighted finite differences over the complete 3x3
    neighbourhood, followed by the Euclidean norm of the gradient vector and
    an arctangent::

        dZ/dx = [(z3 + 2 z6 + z9) - (z1 + 2 z4 + z7)] / (8 dx)
        dZ/di = [(z7 + 2 z8 + z9) - (z1 + 2 z2 + z3)] / (8 dy)
        dZ/dy = orientation * dZ/di

        |grad Z|      = sqrt( (dZ/dx)^2 + (dZ/dy)^2 )
        slope_degrees = arctan(|grad Z|) * 180 / pi

    The focal elevation z5 does not appear: Horn's rule, like the central
    difference, is formed entirely from the cells surrounding the focal
    location. The four DIAGONAL cells z1, z3, z7 and z9 each enter both
    equations with weight 1, while the direct neighbours enter the equation
    for their own direction with weight 2.

    Assumptions
    -----------
    - All nine window cells are valid elevation observations. The caller is
      responsible for the NoData test; this function does not check.
    - Elevation and cell size share the same linear unit, so the derivatives
      are dimensionless.
    - The grid is axis-aligned and regularly spaced.

    Notes
    -----
    Worked example, dx = dy = 10 m, elevations in metres::

        z1=100  z2=102  z3=105
        z4=101  z5=103  z6=106
        z7=104  z8=107  z9=110

        east column weighted sum = z3 + 2 z6 + z9 = 105 + 212 + 110 = 427
        west column weighted sum = z1 + 2 z4 + z7 = 100 + 202 + 104 = 406
        dZ/dx = (427 - 406) / (8 * 10) = 21 / 80 = 0.2625

        bottom row weighted sum  = z7 + 2 z8 + z9 = 104 + 214 + 110 = 428
        top row weighted sum     = z1 + 2 z2 + z3 = 100 + 204 + 105 = 409
        dZ/di = (428 - 409) / (8 * 10) = 19 / 80 = 0.2375
        dZ/dy = -1 * 0.2375 = -0.2375        (north-up raster)

        |grad Z| = sqrt(0.2625^2 + 0.2375^2) = sqrt(0.1253125)
                 = 0.3539950564626574
        theta    = arctan(0.3539950564626574) = 0.34022944683280576 rad
                 = 19.493711369590404 degrees

    The equivalent percent slope is |grad Z| * 100 = 35.40 %, which is the
    same terrain expressed differently, not a different steepness.

    Examples
    --------
    >>> dzdx, dzdy, grad, deg = horn_slope_from_window(
    ...     100, 102, 105, 101, 103, 106, 104, 107, 110, 10.0, 10.0)
    >>> round(dzdx, 10), round(dzdy, 10)
    (0.2625, -0.2375)
    >>> round(grad, 12)
    0.353995056463
    >>> round(deg, 6)
    19.493711

    A plane rising 10 m eastward per 10 m cell and constant north-south, where
    Horn's method and the central difference agree exactly because the surface
    has no curvature transverse to the direction of differentiation:

    >>> dzdx, dzdy, _, deg = horn_slope_from_window(
    ...     0, 10, 20, 0, 10, 20, 0, 10, 20, 10.0, 10.0)
    >>> dzdx, dzdy
    (1.0, -0.0)
    >>> round(deg, 6)
    45.0

    A flat surface has zero slope:

    >>> _, _, grad, deg = horn_slope_from_window(
    ...     5, 5, 5, 5, 5, 5, 5, 5, 5, 2.0, 2.0)
    >>> grad, deg
    (0.0, 0.0)
    """
    # ------------------------------------------------------------------
    # Partial derivative with respect to easting.
    #
    #     dZ/dx = [(z3 + 2 z6 + z9) - (z1 + 2 z4 + z7)] / (8 dx)
    #
    # Numerator   : the weighted elevation of the EAST column minus the
    #               weighted elevation of the WEST column. The direct east and
    #               west neighbours (z6, z4) carry weight 2; the four diagonal
    #               cells (z3, z9, z1, z7) carry weight 1 each.
    # Denominator : 8 dx = (sum of the weights on one side, 4) x (the
    #               separation of each compared pair, 2 dx).
    # Result      : dimensionless rise per unit run in the easting direction,
    #               positive where the terrain rises toward the east.
    #
    # The scale factor converts the nominal map-plane cell size into true
    # ground distance: dividing dx by k is identical to multiplying the
    # quotient by k. k = 1.0 leaves the arithmetic untouched.
    # ------------------------------------------------------------------
    east_weighted = z3 + 2.0 * z6 + z9
    west_weighted = z1 + 2.0 * z4 + z7
    dz_dx = ((east_weighted - west_weighted) / (8.0 * cell_size_x)
             * scale_factor_x)

    # ------------------------------------------------------------------
    # Partial derivative with respect to the ROW INDEX, i.e. in the downward
    # (increasing-row) direction of the array:
    #
    #     dZ/di = [(z7 + 2 z8 + z9) - (z1 + 2 z2 + z3)] / (8 dy)
    #
    # This is positive where elevation increases with increasing row index,
    # which on a north-up raster means rising toward the SOUTH.
    # ------------------------------------------------------------------
    bottom_weighted = z7 + 2.0 * z8 + z9
    top_weighted = z1 + 2.0 * z2 + z3
    dz_di = ((bottom_weighted - top_weighted) / (8.0 * cell_size_y)
             * scale_factor_y)

    # ------------------------------------------------------------------
    # Convert the row-index derivative into the geographic derivative with
    # respect to northing. For a north-up raster the orientation is -1,
    # because moving down a row moves south. The gradient magnitude below
    # squares this term, so the sign cannot change the slope; it is applied
    # so that dz_dy really is the derivative it is documented to be, and so
    # that any later directional product built from these components is not
    # reflected north-for-south.
    # ------------------------------------------------------------------
    dz_dy = y_axis_orientation * dz_di

    # Magnitude of the gradient vector: the maximum rate of elevation change
    # per unit horizontal distance at this cell, in any direction.
    gradient_magnitude = math.sqrt(dz_dx * dz_dx + dz_dy * dz_dy)

    # The gradient magnitude is the tangent of the terrain inclination angle;
    # invert it to recover the angle, then express it in DEGREES.
    slope_degrees = math.degrees(math.atan(gradient_magnitude))

    return dz_dx, dz_dy, gradient_magnitude, slope_degrees


def gradient_magnitude_to_percent(
    gradient_magnitude: np.ndarray,
) -> np.ndarray:
    """
    Convert a gradient magnitude to percent slope.

    Parameters
    ----------
    gradient_magnitude : numpy.ndarray
        |grad Z|, dimensionless (metres of rise per metre of run).

    Returns
    -------
    numpy.ndarray
        Percent slope, ``|grad Z| * 100``.

    Method
    ------
    Percent slope ("percent rise") is the gradient magnitude expressed as a
    percentage rather than converted to an angle::

        slope_percent = |grad Z| * 100
        slope_degrees = arctan(|grad Z|) * 180 / pi

    These are different quantities. Percent slope is unbounded above (a
    vertical face is infinite percent) while slope in degrees is bounded by
    90. They coincide numerically nowhere except at zero::

        |grad Z| = 0.0   ->    0 %   ->   0.00 degrees
        |grad Z| = 0.5   ->   50 %   ->  26.57 degrees
        |grad Z| = 1.0   ->  100 %   ->  45.00 degrees
        |grad Z| = 2.0   ->  200 %   ->  63.43 degrees

    Notes
    -----
    This module writes DEGREES to the output raster. This helper exists only
    so that percent slope can be reported in the summary statistics for
    contrast, keeping the distinction between the two parameterisations
    visible in the run log where it cannot later be conflated.
    """
    return gradient_magnitude * 100.0


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def describe_linear_unit(crs: "CRS") -> Tuple[str, Optional[float]]:
    """
    Report the horizontal linear unit of a CRS.

    Parameters
    ----------
    crs : rasterio.crs.CRS
        Coordinate reference system to interrogate.

    Returns
    -------
    tuple of (str, float or None)
        The unit name (e.g. ``"metre"``), and its size in metres if rasterio
        can determine it (e.g. ``0.3048006096`` for the US survey foot),
        otherwise ``None``.

    Notes
    -----
    The unit is needed for two purposes: to confirm that the cell size is a
    linear distance rather than an angle, and to let the analyst choose a
    correct z-factor when the vertical unit differs from the horizontal unit
    (module docstring, Section 3.2). The lookup is defensive because some CRS
    definitions, particularly hand-written WKT, do not expose a unit factor.
    """
    unit_name = "unknown"
    unit_metres: Optional[float] = None
    try:
        if crs is not None and crs.linear_units:
            unit_name = str(crs.linear_units)
    except Exception:  # pragma: no cover - depends on CRS definition
        pass
    try:
        factor = crs.linear_units_factor
        if factor:
            unit_name = str(factor[0])
            unit_metres = float(factor[1])
    except Exception:  # pragma: no cover - depends on CRS definition
        pass
    return unit_name, unit_metres


def validate_input_raster(
    input_path: str,
    allow_geographic_crs: bool = False,
) -> None:
    """
    Verify that a file is a readable elevation raster fit for Horn's method.

    Parameters
    ----------
    input_path : str
        Filesystem path to the candidate elevation GeoTIFF.
    allow_geographic_crs : bool, optional
        If True, a geographic CRS produces a loud warning instead of an error.
        Default False, which is the scientifically correct behaviour for this
        workflow (module docstring, Section 3.1).

    Returns
    -------
    None
        The function raises on failure and returns nothing on success. It also
        logs the raster's spatial metadata, so that every run records the CRS,
        the CRS type, the x and y resolution, the dimensions and the declared
        NoData value that the computation actually used.

    Raises
    ------
    FileNotFoundError
        The path does not exist or is not a regular file.
    ValueError
        The file cannot be opened as a raster, has no bands, carries a
        non-numeric band type, declares no CRS, declares a geographic CRS
        while ``allow_geographic_crs`` is False, has a rotated or degenerate
        geotransform, is smaller than 3x3 cells, or contains no valid finite
        elevation values.

    Method
    ------
    The checks run from cheapest and most fundamental to most expensive, so
    that an obviously wrong input fails before a full band read:

    1. the path exists and is a file;
    2. the file opens as a raster and has at least one band;
    3. band 1's storage type is real-numeric (not complex, not boolean);
    4. a CRS is declared;
    5. the CRS is projected, not geographic;
    6. the geotransform is axis-aligned (no rotation or shear) and the pixel
       dimensions are finite and strictly positive physical distances;
    7. the raster is at least 3x3 cells, the minimum for one Horn window;
    8. band 1 contains at least one valid finite value, and enough of them for
       a complete 3x3 window to be satisfiable anywhere.

    Assumptions
    -----------
    Band 1 is the elevation surface. A multiband file is accepted with a
    warning; bands 2 and higher are ignored.

    Notes
    -----
    Check 6 rejects rotated ("sheared") geotransforms because Horn's equations
    as implemented assume that array rows and columns are aligned with the CRS
    axes. On a rotated grid, stepping one column is not a step purely in x, so
    the two partial derivatives would each mix both directions and both would
    be wrong. Such a raster must be resampled to an axis-aligned grid first.

    A soft plausibility warning is raised if elevation values fall far outside
    the range of terrestrial elevations in metres. This is a heuristic aid
    only; bathymetry and non-metre vertical units legitimately fall outside
    it, so it never blocks execution.
    """
    # --- 1. Existence -----------------------------------------------------
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input raster does not exist: {input_path}")
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input path is not a file: {input_path}")

    # --- 2. Readability as a raster --------------------------------------
    try:
        dataset = rasterio.open(input_path)
    except Exception as exc:
        raise ValueError(
            f"Input file could not be opened as a raster by rasterio: "
            f"{input_path}\n  underlying error: {exc}"
        ) from exc

    with dataset as src:
        if src.count < 1:
            raise ValueError(f"Raster contains no bands: {input_path}")
        if src.count > 1:
            LOGGER.warning(
                "Raster has %d bands; band 1 is assumed to be the elevation "
                "surface and all other bands are ignored.",
                src.count,
            )

        # --- 3. Band type is real-numeric --------------------------------
        band_dtype = np.dtype(src.dtypes[0])
        if band_dtype.kind not in ("i", "u", "f"):
            raise ValueError(
                f"Band 1 has data type '{band_dtype}', which is not a real "
                "numeric elevation type. An elevation surface must be stored "
                "as an integer or floating-point band."
            )
        if band_dtype.kind in ("i", "u"):
            LOGGER.info(
                "Band 1 is stored as integer type '%s'; it will be cast to "
                "float64 before differencing to avoid integer truncation.",
                band_dtype,
            )

        # --- 4 & 5. CRS presence and type --------------------------------
        if src.crs is None:
            raise ValueError(
                "Raster declares no coordinate reference system. Slope cannot "
                "be computed, because the physical meaning of the cell size "
                "is unknown. Assign or reproject to a projected CRS with "
                "linear units (for example a UTM zone) before running this "
                "script."
            )

        if src.crs.is_geographic:
            message = (
                "The raster uses a GEOGRAPHIC coordinate reference system "
                f"({src.crs.to_string()}), so its cell size is expressed in "
                "DEGREES of angle, not in linear distance.\n"
                "  Horn's denominators 8*dx and 8*dy must be physical "
                "horizontal distances in the same unit as the elevations. "
                "Dividing an elevation difference in metres by a separation "
                "in degrees does not yield a slope: the quotient is not "
                "dimensionless and its arctangent is not a terrain angle.\n"
                "  One degree of longitude also varies with latitude (about "
                "111 km at the equator, about 56 km at 60 degrees, zero at "
                "the poles), so no single cell size in degrees describes the "
                "raster, and degrees of longitude and latitude are not even "
                "equal to each other away from the equator.\n"
                "  Reproject the DEM to a projected CRS with linear units "
                "(for example the appropriate UTM zone, or a local conformal "
                "or equal-area projection for the study area) and re-run."
            )
            if not allow_geographic_crs:
                raise ValueError(message)
            LOGGER.warning(
                "%s\n  allow_geographic_crs was set: continuing, but THE "
                "RESULTING VALUES ARE NOT PHYSICALLY MEANINGFUL SLOPES and "
                "must not be reported as slope angles.",
                message,
            )
        elif not src.crs.is_projected:
            LOGGER.warning(
                "CRS %s is neither clearly projected nor geographic (it may "
                "be a local engineering or unknown CRS). Verify that its "
                "horizontal units are linear distances before trusting the "
                "output.",
                src.crs.to_string(),
            )

        # --- 6. Geotransform is axis-aligned and non-degenerate ----------
        transform = src.transform
        # transform.b and transform.d are the rotation / shear coefficients.
        # They are exactly 0.0 for a conventional north-up, axis-aligned grid.
        if not (math.isclose(transform.b, 0.0, abs_tol=1e-12)
                and math.isclose(transform.d, 0.0, abs_tol=1e-12)):
            raise ValueError(
                "Raster geotransform contains rotation or shear terms "
                f"(b={transform.b}, d={transform.d}). Horn's equations as "
                "implemented here assume rows and columns are aligned with "
                "the CRS axes; on a rotated grid a one-column step is not a "
                "pure step in x. Resample the DEM to an axis-aligned grid "
                "before computing slope."
            )

        cell_size_x = abs(transform.a)
        cell_size_y = abs(transform.e)
        if not (math.isfinite(cell_size_x) and math.isfinite(cell_size_y)):
            raise ValueError(
                f"Raster cell dimensions are not finite "
                f"(dx={transform.a}, dy={transform.e})."
            )
        if cell_size_x <= 0.0 or cell_size_y <= 0.0:
            raise ValueError(
                f"Raster cell dimensions must be non-zero positive distances "
                f"(dx={cell_size_x}, dy={cell_size_y}). A zero cell size "
                "would make the finite-difference denominator zero."
            )

        # --- 7. Raster large enough for a complete Horn window -----------
        if (src.height < MIN_RASTER_DIMENSION
                or src.width < MIN_RASTER_DIMENSION):
            raise ValueError(
                f"Raster is {src.height} rows x {src.width} columns, which is "
                f"too small for Horn's method. At least "
                f"{MIN_RASTER_DIMENSION}x{MIN_RASTER_DIMENSION} cells are "
                "required so that at least one cell has a complete 3x3 "
                "neighbourhood."
            )

        # --- Report the spatial metadata the computation will use --------
        unit_name, _unit_metres = describe_linear_unit(src.crs)
        crs_type = (
            "geographic (angular units)" if src.crs.is_geographic
            else "projected (linear units)" if src.crs.is_projected
            else "unknown"
        )
        LOGGER.info("---- Input raster metadata ----")
        LOGGER.info("Input CRS      : %s", src.crs.to_string())
        LOGGER.info("CRS type       : %s", crs_type)
        LOGGER.info("X resolution   : %.10g %s", cell_size_x, unit_name)
        LOGGER.info("Y resolution   : %.10g %s", cell_size_y, unit_name)
        LOGGER.info("Width          : %d cells", src.width)
        LOGGER.info("Height         : %d cells", src.height)
        LOGGER.info("NoData value   : %s", src.nodata)
        LOGGER.info("Band 1 dtype   : %s", band_dtype)

        if not math.isclose(cell_size_x, cell_size_y, rel_tol=1e-9):
            LOGGER.info(
                "Cells are not square (dx = %.10g, dy = %.10g %s). This is "
                "handled correctly: each Horn derivative is divided by the "
                "cell size of its own axis.",
                cell_size_x, cell_size_y, unit_name,
            )

        # --- 8. Band contains usable elevation values --------------------
        band = src.read(1)
        gdal_valid = src.read_masks(1) != 0
        finite = np.isfinite(band.astype(np.float64, copy=False))
        valid = gdal_valid & finite
        if src.nodata is not None and np.isfinite(src.nodata):
            valid &= band != src.nodata

        valid_count = int(np.count_nonzero(valid))
        if valid_count == 0:
            raise ValueError(
                "Band 1 contains no valid elevation values: every cell is "
                "NoData, NaN or infinite. There is nothing to differentiate."
            )
        if valid_count < HORN_WINDOW_CELLS:
            LOGGER.warning(
                "Band 1 contains only %d valid cells, fewer than the %d "
                "needed to fill a single 3x3 Horn window. The output will "
                "very probably be entirely NoData.",
                valid_count, HORN_WINDOW_CELLS,
            )

        finite_values = band.astype(np.float64, copy=False)[valid]
        z_min = float(np.min(finite_values))
        z_max = float(np.max(finite_values))
        low, high = PLAUSIBLE_ELEVATION_RANGE_M
        if z_min < low or z_max > high:
            LOGGER.warning(
                "Band 1 values span %.3f to %.3f. If these are metres of "
                "elevation they fall outside the usual terrestrial range "
                "(%.0f to %.0f m). Confirm that band 1 really is an elevation "
                "surface, and that its vertical unit matches the horizontal "
                "unit of the CRS (see the z_factor option).",
                z_min, z_max, low, high,
            )


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


def read_elevation_grid(
    input_path: str,
    z_factor: float = 1.0,
    apply_scale_factor_correction: bool = False,
) -> RasterGrid:
    """
    Read an elevation raster and the spatial parameters Horn's method needs.

    Parameters
    ----------
    input_path : str
        Path to a validated elevation GeoTIFF (call
        :func:`validate_input_raster` first).
    z_factor : float, optional
        Multiplicative factor applied to elevation to express it in the same
        linear unit as the horizontal coordinates. Default 1.0, correct when
        both are metres (module docstring, Section 3.2).
    apply_scale_factor_correction : bool, optional
        If True, evaluate the map projection's point scale factor k and divide
        the nominal cell size by it, so that derivatives are taken with
        respect to true ground distance rather than map-plane distance.
        Default False, matching the convention of mainstream GIS slope tools.
        Recommended for ArcticDEM, whose polar stereographic grid carries a
        bias of roughly +4 % at 60 N (module docstring, Section 3.3).

    Returns
    -------
    RasterGrid
        Elevation as float64 with invalid cells set to NaN, the boolean
        validity mask, dx, dy, the y-axis orientation factor, the scale-factor
        fields, and the CRS, transform and profile needed to write a
        co-registered output.

    Raises
    ------
    RuntimeError
        ``apply_scale_factor_correction`` was requested but the correction is
        unavailable, either because ``slope_central_differences.py`` could not
        be imported or because pyproj cannot interrogate the CRS. The error is
        raised rather than silently downgraded, because proceeding without a
        correction that was explicitly requested would leave an undocumented
        systematic bias in the product.

    Method
    ------
    1. Read band 1 and cast to float64 (module docstring, Section 6).
    2. Build the validity mask from three independent sources, combined with
       logical AND so that any one of them can exclude a cell:

       - ``read_masks``, GDAL's own band mask, which already accounts for the
         declared nodata value, an internal mask band and an alpha band;
       - an explicit equality test against ``src.nodata``, which catches a
         nodata value declared in metadata but not honoured by a mask;
       - ``numpy.isfinite``, which excludes NaN and +/- infinity regardless of
         whether they were declared as nodata. NaN in particular cannot be
         caught by equality, since NaN != NaN.

    3. Overwrite every invalid cell with ``numpy.nan``. This is defensive
       redundancy: the validity mask alone is sufficient to exclude those
       cells, but replacing sentinels with NaN guarantees that even a bug in
       the masking logic produces a visible NaN rather than a
       plausible-looking slope derived from a -9999 m "elevation".
    4. Apply the z-factor to the elevations.
    5. Derive dx and dy from the affine transform, and the y-axis orientation
       from the SIGN of the transform's pixel-height term.

    Assumptions
    -----------
    - Band 1 is the elevation surface.
    - The grid is axis-aligned and regularly spaced (enforced by
      :func:`validate_input_raster`).
    - The whole band fits in memory (module docstring, Section 6).

    Notes
    -----
    ``transform.e`` is negative for a conventional north-up raster. Its
    MAGNITUDE is dy; its SIGN determines ``y_axis_orientation``. Taking the
    absolute value for the denominator is what keeps the derivative a rate per
    unit physical distance; the sign is carried separately so that dz_dy is a
    true northward derivative.
    """
    with rasterio.open(input_path) as src:
        raw_band = src.read(1)

        # Cast to float64 before any arithmetic. An integer DEM would
        # otherwise truncate the quotient of the finite difference.
        elevation = raw_band.astype(np.float64, copy=True)

        # --- Validity mask, from three independent criteria ---------------
        valid_mask = src.read_masks(1) != 0
        if src.nodata is not None and np.isfinite(src.nodata):
            valid_mask &= raw_band != src.nodata
        valid_mask &= np.isfinite(elevation)

        # --- Neutralise every invalid cell --------------------------------
        # NaN propagates through arithmetic, so even if a masked cell were
        # mistakenly used, its influence would surface as NaN rather than as
        # a silently wrong slope derived from a sentinel such as -9999.
        elevation[~valid_mask] = np.nan

        # --- Vertical unit conversion -------------------------------------
        if z_factor != 1.0:
            elevation *= z_factor
            LOGGER.info(
                "Applied z-factor %.10g to convert elevation into the "
                "horizontal linear unit of the CRS.",
                z_factor,
            )

        # --- Spatial parameters -------------------------------------------
        transform = src.transform
        cell_size_x = abs(transform.a)
        cell_size_y = abs(transform.e)

        # transform.e < 0 is the north-up case: row index increases southward,
        # so the row-index derivative must be negated to give d/d(northing).
        y_axis_orientation = -1 if transform.e < 0 else 1

        unit_name, _unit_metres = describe_linear_unit(src.crs)
        crs_type = (
            "geographic" if src.crs is not None and src.crs.is_geographic
            else "projected" if src.crs is not None and src.crs.is_projected
            else "unknown"
        )

        # --- Map projection scale factor (Section 3.3) --------------------
        # The cell size read above is a MAP-PLANE distance; the true ground
        # distance is that divided by the point scale factor k. Applying the
        # correction is algebraically identical to multiplying the derivative
        # by k, which is how it is used in the computation.
        scale_x: object = 1.0
        scale_y: object = 1.0
        scale_summary = "not applied (slope computed in nominal grid units)"
        applied = False

        if apply_scale_factor_correction and not _SCALE_FACTOR_AVAILABLE:
            raise RuntimeError(
                "apply_scale_factor_correction was requested, but the "
                "correction could not be loaded from "
                "slope_central_differences.py, which must sit beside this "
                "script in the same folder and which requires pyproj.\n"
                f"  import error: {_SCALE_FACTOR_IMPORT_ERROR}\n"
                "  Install pyproj, restore the companion module, or set "
                "apply_scale_factor_correction to false and accept slope in "
                "nominal grid units."
            )

        if (_SCALE_FACTOR_AVAILABLE and src.crs is not None
                and src.crs.is_projected):
            try:
                (k_x, k_y, summary, max_departure,
                 mean_k) = compute_scale_factor_field(
                    src.crs, transform, (src.height, src.width)
                )
            except RuntimeError as exc:
                if apply_scale_factor_correction:
                    raise
                LOGGER.debug("Scale factor not evaluated: %s", exc)
            else:
                if apply_scale_factor_correction:
                    scale_x, scale_y = k_x, k_y
                    scale_summary = f"applied: {summary}"
                    applied = True
                    LOGGER.info(
                        "Map projection scale-factor correction APPLIED: %s. "
                        "Derivatives are taken with respect to true ground "
                        "distance, not map-plane distance.",
                        summary,
                    )
                elif max_departure > SCALE_FACTOR_WARN_THRESHOLD:
                    # Off by default, but never silently: state the exact cost.
                    LOGGER.warning(
                        "Map projection distortion is significant for this "
                        "raster: %s.\n"
                        "  Slope is being computed in NOMINAL grid units, so "
                        "every gradient is biased by up to %.2f %% and the "
                        "resulting angles are systematically %s.\n"
                        "  Enable apply_scale_factor_correction (CLI: "
                        "--apply-scale-factor) to divide the cell size by the "
                        "point scale factor and remove this bias.",
                        summary,
                        100.0 * max_departure,
                        # k > 1 means map distances exceed ground distances,
                        # so the nominal denominator is too large and the
                        # gradient - hence the angle - comes out too small.
                        "understated" if mean_k > 1.0 else "overstated",
                    )

        return RasterGrid(
            elevation=elevation,
            valid_mask=valid_mask,
            cell_size_x=cell_size_x,
            cell_size_y=cell_size_y,
            y_axis_orientation=y_axis_orientation,
            transform=transform,
            crs=src.crs,
            linear_unit=unit_name,
            crs_type=crs_type,
            nodata_value=(None if src.nodata is None else float(src.nodata)),
            profile=src.profile.copy(),
            scale_factor_x=scale_x,
            scale_factor_y=scale_y,
            scale_factor_applied=applied,
            scale_factor_summary=scale_summary,
        )


# --------------------------------------------------------------------------
# Core computation
# --------------------------------------------------------------------------


def calculate_horn_slope(grid: RasterGrid) -> SlopeResult:
    """
    Compute slope in degrees by Horn's method over the whole raster.

    Parameters
    ----------
    grid : RasterGrid
        Validated elevation grid from :func:`read_elevation_grid`.

    Returns
    -------
    SlopeResult
        Slope in degrees, the two partial derivatives, the gradient magnitude,
        and the boolean mask of cells that received a defensible value. Every
        non-evaluable cell is ``numpy.nan`` in all four arrays.

    Method
    ------
    Horn's (1981) weighted finite differences over the complete 3x3
    neighbourhood, applied to the whole array at once by NumPy slicing.

    The nine aligned views into the array are named for their position
    relative to the focal cell, and correspond exactly to the z1..z9 labels of
    the module docstring, Section 1.2::

        z1 z2 z3        [i-1, j-1]  [i-1, j]  [i-1, j+1]      row above
        z4 z5 z6   =    [i,   j-1]  [i,   j]  [i,   j+1]      focal row
        z7 z8 z9        [i+1, j-1]  [i+1, j]  [i+1, j+1]      row below

    All nine views have identical shape (rows-2, cols-2) and are aligned
    cell-for-cell, so element k of one corresponds to element k of every
    other. Slicing creates VIEWS, not copies, so the nine "neighbourhoods"
    cost no additional memory; only the arithmetic allocates.

    The derivatives are then::

        dZ/dx = [(z3 + 2 z6 + z9) - (z1 + 2 z4 + z7)] / (8 dx)
        dZ/di = [(z7 + 2 z8 + z9) - (z1 + 2 z2 + z3)] / (8 dy)
        dZ/dy = orientation * dZ/di

        |grad Z|      = sqrt( (dZ/dx)^2 + (dZ/dy)^2 )
        slope_degrees = arctan(|grad Z|) * 180 / pi

    No Python loop iterates over cells: a loop over a raster of even modest
    size would be several hundred times slower than the vectorised form, with
    no gain in clarity, since the slicing expressions state the stencil more
    directly than nested index arithmetic would.

    Assumptions
    -----------
    - The grid is axis-aligned and regularly spaced.
    - Elevation and cell size share a linear unit.
    - Elevation, validity mask and any scale-factor field share one shape.

    Notes
    -----
    Output arrays are allocated pre-filled with NaN, so that "not evaluated"
    is the DEFAULT state of every cell and a value has to be earned by passing
    the 9-cell validity test. The boundary ring is therefore never written to
    and needs no special-casing: it simply keeps its NaN (module docstring,
    Section 5).
    """
    elevation = grid.elevation
    valid = grid.valid_mask
    rows, cols = elevation.shape
    dx = grid.cell_size_x
    dy = grid.cell_size_y

    if rows < MIN_RASTER_DIMENSION or cols < MIN_RASTER_DIMENSION:
        raise ValueError(
            f"Grid is {rows} x {cols}; Horn's method needs at least "
            f"{MIN_RASTER_DIMENSION} x {MIN_RASTER_DIMENSION} cells."
        )

    LOGGER.info(
        "Computing Horn 3x3 finite differences on a %d x %d grid "
        "(dx = %.10g, dy = %.10g %s, y-axis orientation = %+d).",
        rows, cols, dx, dy, grid.linear_unit, grid.y_axis_orientation,
    )

    # Allocate output arrays pre-filled with NaN. Every cell the method cannot
    # legitimately evaluate - the boundary ring, and any cell with an
    # incomplete 3x3 window - simply keeps its NaN.
    dz_dx = np.full((rows, cols), np.nan, dtype=np.float64)
    dz_dy = np.full((rows, cols), np.nan, dtype=np.float64)

    # ------------------------------------------------------------------
    # The nine aligned slices of the 3x3 neighbourhood.
    #
    # Rows:    [ :-2] is the row ABOVE the focal row (i-1, north on a
    #                 north-up raster)
    #          [1:-1] is the focal row (i)
    #          [2:  ] is the row BELOW the focal row (i+1, south)
    # Columns: [ :-2] is the column WEST of the focal column (j-1)
    #          [1:-1] is the focal column (j)
    #          [2:  ] is the column EAST of the focal column (j+1)
    #
    # Each is a view of shape (rows-2, cols-2); together they cover every
    # focal cell that has a complete 3x3 window, which is exactly the
    # interior of the raster.
    # ------------------------------------------------------------------
    above, focal_row, below = slice(None, -2), slice(1, -1), slice(2, None)
    west, focal_col, east = slice(None, -2), slice(1, -1), slice(2, None)

    z1 = elevation[above, west]        # north-west diagonal
    z2 = elevation[above, focal_col]   # north
    z3 = elevation[above, east]        # north-east diagonal
    z4 = elevation[focal_row, west]    # west
    z5 = elevation[focal_row, focal_col]   # FOCAL cell (validity only)
    z6 = elevation[focal_row, east]    # east
    z7 = elevation[below, west]        # south-west diagonal
    z8 = elevation[below, focal_col]   # south
    z9 = elevation[below, east]        # south-east diagonal

    interior = (focal_row, focal_col)

    # ------------------------------------------------------------------
    # Window completeness test.
    #
    # A slope value is computed only where ALL NINE cells of the window hold
    # genuine elevation observations. The logical AND of the nine aligned
    # validity slices is exactly that condition (module docstring, Section 4).
    #
    # z5 contributes nothing to the arithmetic but is required here, so that
    # no slope value is written at a location where the DEM records no ground.
    # ------------------------------------------------------------------
    v1 = valid[above, west]
    v2 = valid[above, focal_col]
    v3 = valid[above, east]
    v4 = valid[focal_row, west]
    v5 = valid[focal_row, focal_col]
    v6 = valid[focal_row, east]
    v7 = valid[below, west]
    v8 = valid[below, focal_col]
    v9 = valid[below, east]

    window_valid = np.zeros((rows, cols), dtype=bool)
    window_valid[interior] = (
        v1 & v2 & v3 & v4 & v5 & v6 & v7 & v8 & v9
    )

    excluded_by_nodata = int(
        np.count_nonzero(valid[interior] & ~window_valid[interior])
    )
    if excluded_by_nodata:
        LOGGER.info(
            "%d interior cells hold valid elevation but have at least one "
            "NoData cell in their 3x3 neighbourhood; they are assigned "
            "NoData in the slope output rather than being estimated from an "
            "incomplete window.",
            excluded_by_nodata,
        )

    # The scale-factor correction (module docstring, Section 3.3) divides the
    # nominal cell size by the point scale factor k to obtain the true ground
    # cell size. Dividing the denominator by k is identical to multiplying the
    # quotient by k, which is the form used here:
    #
    #     dZ/dx = [...] / (8 * dx / k)  =  k * [...] / (8 * dx)
    #
    # k is exactly 1.0 when the correction is disabled, leaving the arithmetic
    # unchanged. Where k is a field it is sliced to the interior to stay
    # aligned with the derivative arrays.
    k_x = grid.scale_factor_x
    k_y = grid.scale_factor_y
    k_x_interior = k_x[interior] if isinstance(k_x, np.ndarray) else k_x
    k_y_interior = k_y[interior] if isinstance(k_y, np.ndarray) else k_y

    # ------------------------------------------------------------------
    # Partial derivative with respect to easting (Horn's x equation):
    #
    #                (z3 + 2 z6 + z9) - (z1 + 2 z4 + z7)
    #     dZ/dx  =  -------------------------------------
    #                              8 dx
    #
    # Numerator   : the weighted elevation of the EAST column minus that of
    #               the WEST column. The direct east and west neighbours
    #               (z6, z4) carry weight 2; the four diagonal cells
    #               (z3, z9 east, z1, z7 west) carry weight 1 each.
    # Denominator : 8 dx = (weights on one side, 1+2+1 = 4) x (separation of
    #               each compared pair, 2 dx). Equivalently, the expression is
    #               the 1:2:1 weighted mean of the three row-wise central
    #               differences (z3-z1)/2dx, (z6-z4)/2dx and (z9-z7)/2dx.
    # Result      : dimensionless rise per unit run toward the east.
    # ------------------------------------------------------------------
    east_weighted = z3 + 2.0 * z6 + z9
    west_weighted = z1 + 2.0 * z4 + z7
    dz_dx[interior] = (
        (east_weighted - west_weighted) / (8.0 * dx) * k_x_interior
    )

    # ------------------------------------------------------------------
    # Partial derivative with respect to the ROW INDEX (Horn's y equation as
    # published):
    #
    #                (z7 + 2 z8 + z9) - (z1 + 2 z2 + z3)
    #     dZ/di  =  -------------------------------------
    #                              8 dy
    #
    # Positive where elevation increases with increasing row index, which on a
    # north-up raster means rising toward the SOUTH.
    # ------------------------------------------------------------------
    bottom_weighted = z7 + 2.0 * z8 + z9
    top_weighted = z1 + 2.0 * z2 + z3
    dz_di = (
        (bottom_weighted - top_weighted) / (8.0 * dy) * k_y_interior
    )

    # ------------------------------------------------------------------
    # Convert the row-index derivative into the geographic derivative with
    # respect to northing:
    #
    #     dZ/dy = orientation * dZ/di,   orientation = -1 for a north-up grid
    #
    # The gradient magnitude below squares this term, so the sign cannot
    # change the slope (module docstring, Section 2.2). It is applied because
    # dz_dy is documented and returned as the derivative with respect to
    # northing and must therefore actually be that, and because any later
    # signed use - aspect, hillshade, curvature, flow routing - depends on it.
    # ------------------------------------------------------------------
    dz_dy[interior] = grid.y_axis_orientation * dz_di

    # `z5` participates in the validity test only; naming it here documents
    # that its exclusion from the arithmetic is deliberate, not an oversight.
    del z5

    # ------------------------------------------------------------------
    # Gradient magnitude:
    #
    #     |grad Z| = sqrt( (dZ/dx)^2 + (dZ/dy)^2 )
    #
    # The Euclidean norm of the gradient vector: the maximum rate of elevation
    # change per unit horizontal distance at the cell, attained in the
    # direction of steepest ascent (whose azimuth is the aspect, not computed
    # here). Dimensionless, and equal to the tangent of the terrain angle.
    # numpy.hypot is used rather than sqrt(a*a + b*b) because it avoids
    # intermediate overflow and underflow.
    # ------------------------------------------------------------------
    gradient_magnitude = np.hypot(dz_dx, dz_dy)

    # ------------------------------------------------------------------
    # Slope angle:
    #
    #     theta         = arctan(|grad Z|)           [radians]
    #     theta_degrees = theta * 180 / pi           [DEGREES]
    #
    # arctan maps [0, inf) onto [0, 90) degrees, so the result is bounded: a
    # finite gradient can never produce exactly 90 degrees, consistent with
    # the fact that a single-valued surface Z(x, y) cannot represent a
    # vertical face.
    # ------------------------------------------------------------------
    slope_degrees = np.degrees(np.arctan(gradient_magnitude))

    # ------------------------------------------------------------------
    # Final sanitisation. The window test above should already exclude every
    # cell that could yield a non-finite result, but the output mask is
    # additionally intersected with a finiteness test, so that no NaN or
    # infinity can reach the written file whatever the input contained.
    # ------------------------------------------------------------------
    output_valid_mask = window_valid & np.isfinite(slope_degrees)
    leaked = int(np.count_nonzero(window_valid & ~output_valid_mask))
    if leaked:
        LOGGER.warning(
            "%d cells passed the 3x3 window validity test but produced a "
            "non-finite slope; they have been demoted to NoData. This "
            "indicates unusual input values and should be investigated "
            "rather than accepted.",
            leaked,
        )

    slope_degrees[~output_valid_mask] = np.nan
    dz_dx[~output_valid_mask] = np.nan
    dz_dy[~output_valid_mask] = np.nan
    gradient_magnitude[~output_valid_mask] = np.nan

    return SlopeResult(
        slope_degrees=slope_degrees,
        dz_dx=dz_dx,
        dz_dy=dz_dy,
        gradient_magnitude=gradient_magnitude,
        output_valid_mask=output_valid_mask,
    )


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def print_statistics(grid: RasterGrid, result: SlopeResult) -> dict:
    """
    Compute, log and return quality-control statistics for the slope raster.

    Parameters
    ----------
    grid : RasterGrid
        The input elevation grid, for the count of valid input cells.
    result : SlopeResult
        The computed slope result.

    Returns
    -------
    dict
        Statistics suitable for a run log or for embedding in the output
        metadata: cell counts and percentages, and the minimum, maximum,
        mean, median and standard deviation of the valid slope values in
        degrees, plus the mean expressed as percent slope for contrast.

    Method
    ------
    Statistics are computed over the VALID output cells only. Including
    NoData cells, whatever sentinel they carry, would corrupt every moment: a
    single -9999 would drag the mean below zero. The median is computed
    exactly with ``numpy.median`` rather than approximated; on a raster of any
    realistic size this is an O(n) selection and is cheap relative to the
    raster read.

    The result is then audited for values that are mathematically impossible
    for a correctly computed arctangent:

    - NaN, which would mean an invalid cell escaped the validity mask;
    - +/- infinity, which would mean an overflow;
    - values below 0 or at or above 90 degrees, which arctan cannot produce
      from a finite gradient.

    Any such value indicates a defect in the implementation or a corrupted
    input, not an unusual landscape. They are reported loudly, with their
    count and extreme values, and are NEVER silently clipped into range:
    clipping would hide the defect while leaving the product wrong. The
    diagnosis to pursue is, in order, the DEM's NoData declaration, the
    z-factor, and the cell size - a slope pinned near 90 degrees over a wide
    area is the signature of a sentinel value entering the arithmetic.

    Notes
    -----
    The count of valid output cells is expected to be smaller than the count
    of valid input cells even on a gap-free DEM, because the boundary ring is
    always excluded (module docstring, Section 5). The difference is
    decomposed in the log into the boundary contribution and the
    NoData-dilation contribution, so that an unexpectedly large loss is
    visible immediately.

    The mean slope in degrees is not the arctangent of the mean gradient, and
    the mean of a set of angles is not in general a meaningful summary of
    terrain. These statistics are reported for quality control - detecting an
    all-flat or all-vertical result, for instance - not as terrain parameters
    to be published without further thought.
    """
    rows, cols = grid.elevation.shape
    total_cells = rows * cols

    valid_input = int(np.count_nonzero(grid.valid_mask))
    valid_output = int(np.count_nonzero(result.output_valid_mask))
    nodata_output = total_cells - valid_output

    # Boundary ring: the outermost row and column on each side.
    boundary_cells = total_cells - (rows - 2) * (cols - 2)

    values = result.slope_degrees[result.output_valid_mask]

    stats = {
        "raster_rows": rows,
        "raster_cols": cols,
        "total_cells": total_cells,
        "valid_input_cells": valid_input,
        "nodata_input_cells": total_cells - valid_input,
        "valid_output_cells": valid_output,
        "nodata_output_cells": nodata_output,
        "valid_output_percent": 100.0 * valid_output / total_cells,
        "nodata_output_percent": 100.0 * nodata_output / total_cells,
        "boundary_cells_excluded": boundary_cells,
        "nodata_dilation_cells": max(nodata_output - boundary_cells, 0),
    }

    if valid_output > 0:
        gradients = result.gradient_magnitude[result.output_valid_mask]
        stats.update(
            {
                "slope_min_degrees": float(np.min(values)),
                "slope_max_degrees": float(np.max(values)),
                "slope_mean_degrees": float(np.mean(values)),
                "slope_median_degrees": float(np.median(values)),
                "slope_stddev_degrees": float(np.std(values)),
                "slope_mean_percent": float(
                    np.mean(gradient_magnitude_to_percent(gradients))
                ),
            }
        )

    LOGGER.info("---- Validation and quality control ----")
    LOGGER.info(
        "Raster dimensions             : %d rows x %d cols (%d cells)",
        rows, cols, total_cells,
    )
    LOGGER.info(
        "Valid elevation cells (input) : %d (%.2f%% of raster)",
        valid_input, 100.0 * valid_input / total_cells,
    )
    LOGGER.info(
        "NoData cells (input)          : %d (%.2f%% of raster)",
        stats["nodata_input_cells"],
        100.0 * stats["nodata_input_cells"] / total_cells,
    )
    LOGGER.info(
        "Valid slope cells (output)    : %d (%.2f%% of raster)",
        valid_output, stats["valid_output_percent"],
    )
    LOGGER.info(
        "NoData cells (output)         : %d (%.2f%% of raster)",
        nodata_output, stats["nodata_output_percent"],
    )
    LOGGER.info(
        "  of which boundary ring      : %d (3x3 window incomplete at edges)",
        boundary_cells,
    )
    LOGGER.info(
        "  of which NoData-adjacent    : %d (incomplete 9-cell window)",
        stats["nodata_dilation_cells"],
    )

    if valid_output > 0:
        LOGGER.info("Slope minimum                 : %.6f degrees",
                    stats["slope_min_degrees"])
        LOGGER.info("Slope maximum                 : %.6f degrees",
                    stats["slope_max_degrees"])
        LOGGER.info("Slope mean                    : %.6f degrees",
                    stats["slope_mean_degrees"])
        LOGGER.info("Slope median                  : %.6f degrees",
                    stats["slope_median_degrees"])
        LOGGER.info("Slope standard deviation      : %.6f degrees",
                    stats["slope_stddev_degrees"])
        LOGGER.info("Mean gradient as percent slope: %.4f %% "
                    "(reported for contrast; the raster stores DEGREES)",
                    stats["slope_mean_percent"])
    else:
        LOGGER.warning(
            "No cell received a valid slope value. Every cell was either on "
            "the raster boundary or had an incomplete 3x3 neighbourhood. "
            "Check the extent of NoData in the input DEM."
        )

    # --- Audit for numerically impossible results ------------------------
    full = result.slope_degrees
    nan_count = int(
        np.count_nonzero(np.isnan(full) & result.output_valid_mask)
    )
    posinf_count = int(np.count_nonzero(np.isposinf(full)))
    neginf_count = int(np.count_nonzero(np.isneginf(full)))
    out_of_range = 0
    if valid_output > 0:
        out_of_range = int(
            np.count_nonzero(
                (values < 0.0) | (values >= MAX_ATTAINABLE_SLOPE_DEGREES)
            )
        )

    stats.update(
        {
            "nan_cells_in_valid_area": nan_count,
            "positive_infinity_cells": posinf_count,
            "negative_infinity_cells": neginf_count,
            "out_of_range_cells": out_of_range,
        }
    )

    if nan_count or posinf_count or neginf_count or out_of_range:
        LOGGER.error(
            "NUMERICAL ANOMALY DETECTED in the computed slope: "
            "NaN in the valid area = %d, +inf = %d, -inf = %d, values outside "
            "[0, 90) degrees = %d.\n"
            "  These results are mathematically impossible for an arctangent "
            "of a finite gradient, so they indicate a defect rather than an "
            "unusual landscape. They have NOT been clipped. Investigate, in "
            "order: the NoData declaration of the input DEM (a sentinel "
            "entering the arithmetic pins slope near 90 degrees), the "
            "z-factor, and the cell size.",
            nan_count, posinf_count, neginf_count, out_of_range,
        )
    else:
        LOGGER.info(
            "Numerical audit               : clean (no NaN or infinity in the "
            "valid area; every value within the attainable range [0, 90) "
            "degrees)."
        )

    return stats


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def save_slope_raster(
    output_path: str,
    result: SlopeResult,
    grid: RasterGrid,
    stats: dict,
    input_path: str,
    z_factor: float,
    nodata_value: float = DEFAULT_OUTPUT_NODATA,
) -> None:
    """
    Write the slope raster as a GeoTIFF co-registered with the input DEM.

    Parameters
    ----------
    output_path : str
        Destination path for the slope GeoTIFF.
    result : SlopeResult
        The computed slope, whose ``slope_degrees`` array is written.
    grid : RasterGrid
        The source grid, supplying the CRS, transform and profile that are
        preserved in the output.
    stats : dict
        Statistics from :func:`print_statistics`, a subset of which is
        embedded as GeoTIFF metadata tags for provenance.
    input_path : str
        Path of the source DEM, recorded as provenance metadata.
    z_factor : float
        The z-factor that was applied, recorded as provenance metadata.
    nodata_value : float, optional
        Value written for cells with no defensible slope. Default -9999.0.

    Returns
    -------
    None

    Method
    ------
    The output profile is derived from the INPUT profile, so that spatial
    extent, CRS, affine transform and raster dimensions are preserved exactly
    and the slope raster overlays the DEM cell for cell. Only the properties
    that must change are overridden:

    - ``count`` is set to 1 (slope is a single scalar surface);
    - ``dtype`` is set to float32 (module docstring, Section 6);
    - ``nodata`` is set to the output sentinel, which lies outside the
      attainable range [0, 90) of slope in degrees and so can never be
      confused with a real value;
    - LZW compression and tiling are enabled, both lossless: they change no
      pixel value, only how the bytes are arranged on disk.

    Immediately before writing, every invalid cell is replaced by the NoData
    sentinel and the array is asserted to be entirely finite, so that NaN and
    infinity cannot be written to the file.

    Preserving the transform and CRS is not a formality. Every later step of
    the workflow - stacking slope with spectral predictors, extracting
    training samples, validating against field points, mosaicking tiles -
    assumes that a given cell of the slope raster refers to the same patch of
    ground as the corresponding cell of the DEM. A slope raster with a
    different transform would introduce a silent spatial offset into all of
    them, and at 1 m resolution a half-cell shift is enough to sample the
    wrong side of a terrain break.

    Notes
    -----
    Metadata tags record only facts that are true of this product: the derived
    quantity, the numerical method, the units, the neighbourhood, the boundary
    and NoData rules, the cell size actually used, the z-factor, whether the
    scale-factor correction was applied, and the source file. No tag asserts
    equivalence with any GIS package's slope tool, and none records a quantity
    the file does not contain.
    """
    slope = result.slope_degrees

    # Substitute the sentinel for every cell without a defensible value. This
    # is the only place a NoData value is introduced into the slope array.
    output_array = np.where(result.output_valid_mask, slope, nodata_value)

    # Hard guarantee that no NaN or infinity is written to disk.
    if not np.all(np.isfinite(output_array)):
        raise ValueError(
            "Internal consistency failure: the output array still contains "
            "non-finite values after NoData substitution. Refusing to write "
            "a corrupted raster."
        )

    output_array = output_array.astype(OUTPUT_DTYPE, copy=False)

    profile = grid.profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype=OUTPUT_DTYPE,
        nodata=nodata_value,
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    # Extent, CRS, transform and dimensions are inherited unchanged from the
    # input profile; restate them explicitly so the guarantee is visible here
    # rather than implied by the copy above.
    profile.update(
        width=grid.profile["width"],
        height=grid.profile["height"],
        transform=grid.transform,
        crs=grid.crs,
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(output_array, 1)

        # Band-level description, shown by gdalinfo and by most GIS clients.
        dst.set_band_description(1, "Slope angle (degrees), Horn 3x3 method")

        # Dataset-level provenance tags. Every entry is a verifiable statement
        # about how this specific file was produced.
        dst.update_tags(
            derived_product="Slope",
            input_type="Elevation raster",
            input_variable="elevation",
            method="Horn finite-difference method",
            finite_difference_method="Horn (1981) 3x3 weighted",
            finite_difference_stencil=(
                "complete 3x3 neighbourhood; orthogonal neighbours weight 2, "
                "diagonal neighbours weight 1"
            ),
            dz_dx_equation="[(z3 + 2*z6 + z9) - (z1 + 2*z4 + z7)] / (8*dx)",
            dz_dy_equation="[(z7 + 2*z8 + z9) - (z1 + 2*z2 + z3)] / (8*dy)",
            gradient_definition="sqrt((dZ/dx)^2 + (dZ/dy)^2)",
            slope_definition="arctan(gradient_magnitude) * 180 / pi",
            units="degrees",
            slope_units="degrees",
            slope_value_range="[0, 90)",
            cell_size_x=f"{grid.cell_size_x:.10g}",
            cell_size_y=f"{grid.cell_size_y:.10g}",
            horizontal_linear_unit=grid.linear_unit,
            crs_type=grid.crs_type,
            input_nodata=str(grid.nodata_value),
            z_factor=f"{z_factor:.10g}",
            # Whether the cell size used was the nominal map-plane size or was
            # corrected to true ground distance. This materially changes the
            # values, so it is recorded unambiguously in the product itself.
            scale_factor_correction=(
                "applied" if grid.scale_factor_applied else "not applied"
            ),
            scale_factor_detail=grid.scale_factor_summary,
            horizontal_distance_basis=(
                "true ground distance (projection scale factor removed)"
                if grid.scale_factor_applied
                else "nominal map-plane distance (projection scale factor "
                     "not removed)"
            ),
            boundary_treatment="Outer one-cell ring assigned NoData "
                               "(3x3 window incomplete at raster edges)",
            nodata_treatment="Cell assigned NoData unless all nine cells of "
                             "its 3x3 neighbourhood are valid",
            computation_precision="float64 internally, float32 storage",
            source_dem=os.path.basename(input_path),
            valid_output_cells=str(stats.get("valid_output_cells", "")),
            nodata_output_cells=str(stats.get("nodata_output_cells", "")),
            reference="Horn, B. K. P. (1981). Hill shading and the "
                      "reflectance map. Proc. IEEE 69(1), 14-47.",
            software="Horns.py (rasterio + numpy)",
        )

    LOGGER.info("Slope raster written: %s", output_path)
    LOGGER.info(
        "  driver=GTiff  dtype=%s  nodata=%.1f  size=%d x %d  crs=%s  "
        "units=degrees",
        OUTPUT_DTYPE, nodata_value,
        grid.profile["height"], grid.profile["width"],
        grid.crs.to_string(),
    )


# --------------------------------------------------------------------------
# Configuration file
# --------------------------------------------------------------------------


#: Keys permitted in a configuration file job (and in ``defaults``). Any other
#: key is an error rather than a silent no-op, so that a typo such as
#: ``input_rasters`` cannot cause the wrong file to be processed unnoticed.
CONFIG_JOB_KEYS = frozenset({
    "name",
    "input_raster",
    "output_raster",
    "z_factor",
    "output_nodata",
    "overwrite",
    "apply_scale_factor_correction",
    "allow_geographic_crs",
})


@dataclass(frozen=True)
class HornJob:
    """
    One fully resolved Horn slope-derivation task.

    Attributes
    ----------
    name : str
        Label for logs and error messages, taken from the configuration file
        or derived from the input filename.
    input_raster : str
        Absolute path to the elevation raster.
    output_raster : str
        Absolute path for the slope raster.
    z_factor : float
        Vertical-to-horizontal unit conversion (module docstring, Section 3.2).
    output_nodata : float
        NoData sentinel for the output.
    overwrite : bool
        Whether an existing output may be replaced.
    apply_scale_factor_correction : bool
        Whether to correct for map projection distortion (Section 3.3).
    allow_geographic_crs : bool
        Whether to proceed despite a geographic CRS. Not recommended.
    """

    name: str
    input_raster: str
    output_raster: str
    z_factor: float = 1.0
    output_nodata: float = DEFAULT_OUTPUT_NODATA
    overwrite: bool = False
    apply_scale_factor_correction: bool = False
    allow_geographic_crs: bool = False


def load_config(config_path: str) -> list:
    """
    Read a YAML configuration file and resolve it into a list of jobs.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file, normally
        ``config/horn_config.yaml``.

    Returns
    -------
    list of HornJob
        One entry per job defined in the file, in file order, with the
        ``defaults`` block merged into each and all paths resolved.

    Raises
    ------
    RuntimeError
        PyYAML is not installed.
    FileNotFoundError
        The configuration file does not exist.
    ValueError
        The file is not valid YAML, is not a mapping, defines no jobs, uses an
        unrecognised key, omits a required path, or gives a value of the wrong
        type.

    Method
    ------
    The file is a mapping that may contain two keys:

    - ``defaults``: settings applied to every job, and
    - ``jobs``: a list of job mappings, each of which may override any default.

    A file with neither key is treated as a single job defined at the top
    level, so the simplest usable configuration is just an ``input_raster``
    and an ``output_raster``.

    Relative paths are resolved against the DIRECTORY CONTAINING THE
    CONFIGURATION FILE, not the current working directory. This makes a
    configuration reproducible: it can be committed alongside the data and run
    from anywhere without rewriting, and the same file produces the same
    result on a collaborator's machine.

    Unrecognised keys raise an error rather than being ignored. In a research
    workflow a silently-dropped key is a correctness hazard: a misspelled
    ``z_factor`` would leave the elevation unconverted, and a misspelled
    ``apply_scale_factor_correction`` would silently leave a systematic bias
    in the product.

    Notes
    -----
    The format is identical to that of ``slope_config.yaml``, which drives the
    central-difference module, so the two are interchangeable in shape and a
    reader who knows one knows the other. They are kept in SEPARATE FILES
    because each names the output path of its own product: a single file
    cannot hold two different outputs for one input without ambiguity, and
    strict key validation means a key added for one method would be rejected
    by the other.

    YAML is parsed with ``yaml.safe_load``, which constructs only plain Python
    scalars and containers, so a configuration file cannot instantiate
    arbitrary objects.
    """
    if not _YAML_AVAILABLE:
        raise RuntimeError(
            "Reading a configuration file requires 'PyYAML', which is not "
            "installed. Install it with 'pip install pyyaml', or pass the "
            "input and output paths as positional arguments instead."
        )

    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"Configuration file does not exist: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as handle:
        try:
            document = yaml.safe_load(handle)
        except yaml.YAMLError as exc:
            raise ValueError(
                f"Configuration file is not valid YAML: {config_path}\n"
                f"  {exc}"
            ) from exc

    if document is None:
        raise ValueError(f"Configuration file is empty: {config_path}")
    if not isinstance(document, dict):
        raise ValueError(
            f"Configuration file must contain a mapping at the top level, "
            f"got {type(document).__name__}: {config_path}"
        )

    unknown_top = set(document) - {"defaults", "jobs"} - CONFIG_JOB_KEYS
    if unknown_top:
        raise ValueError(
            f"Unrecognised key(s) in {config_path}: "
            f"{', '.join(sorted(unknown_top))}\n"
            f"  Permitted top-level keys: defaults, jobs, or a single job's "
            f"keys ({', '.join(sorted(CONFIG_JOB_KEYS))})."
        )

    defaults = document.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise ValueError(
            f"'defaults' must be a mapping in {config_path}, "
            f"got {type(defaults).__name__}."
        )
    unknown_defaults = set(defaults) - CONFIG_JOB_KEYS
    if unknown_defaults:
        raise ValueError(
            f"Unrecognised key(s) under 'defaults' in {config_path}: "
            f"{', '.join(sorted(unknown_defaults))}"
        )

    raw_jobs = document.get("jobs")
    if raw_jobs is None:
        # No 'jobs' list: treat the top-level mapping itself as one job.
        single = {k: v for k, v in document.items() if k in CONFIG_JOB_KEYS}
        if not single:
            raise ValueError(
                f"Configuration file defines no jobs: {config_path}\n"
                "  Provide a 'jobs:' list, or 'input_raster' and "
                "'output_raster' at the top level."
            )
        raw_jobs = [single]
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise ValueError(f"'jobs' must be a non-empty list in {config_path}.")

    config_dir = os.path.dirname(os.path.abspath(config_path))

    def _resolve(path_value: str) -> str:
        """Resolve a path relative to the configuration file's directory."""
        expanded = os.path.expanduser(str(path_value))
        if os.path.isabs(expanded):
            return os.path.normpath(expanded)
        return os.path.normpath(os.path.join(config_dir, expanded))

    jobs = []
    for index, raw in enumerate(raw_jobs, start=1):
        if not isinstance(raw, dict):
            raise ValueError(
                f"Job {index} in {config_path} must be a mapping, "
                f"got {type(raw).__name__}."
            )
        unknown = set(raw) - CONFIG_JOB_KEYS
        if unknown:
            raise ValueError(
                f"Unrecognised key(s) in job {index} of {config_path}: "
                f"{', '.join(sorted(unknown))}\n"
                f"  Permitted keys: {', '.join(sorted(CONFIG_JOB_KEYS))}"
            )

        merged = dict(defaults)
        merged.update(raw)

        label = str(merged.get("name") or f"job_{index}")

        for required in ("input_raster", "output_raster"):
            if not merged.get(required):
                raise ValueError(
                    f"Job '{label}' in {config_path} is missing required key "
                    f"'{required}'."
                )

        try:
            z_factor = float(merged.get("z_factor", 1.0))
            output_nodata = float(
                merged.get("output_nodata", DEFAULT_OUTPUT_NODATA)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Job '{label}' in {config_path} has a non-numeric "
                f"'z_factor' or 'output_nodata': {exc}"
            ) from exc

        for flag in ("overwrite", "apply_scale_factor_correction",
                     "allow_geographic_crs"):
            value = merged.get(flag, False)
            if not isinstance(value, bool):
                raise ValueError(
                    f"Job '{label}' in {config_path}: '{flag}' must be true "
                    f"or false, got {value!r}."
                )

        jobs.append(
            HornJob(
                name=label,
                input_raster=_resolve(merged["input_raster"]),
                output_raster=_resolve(merged["output_raster"]),
                z_factor=z_factor,
                output_nodata=output_nodata,
                overwrite=bool(merged.get("overwrite", False)),
                apply_scale_factor_correction=bool(
                    merged.get("apply_scale_factor_correction", False)
                ),
                allow_geographic_crs=bool(
                    merged.get("allow_geographic_crs", False)
                ),
            )
        )

    return jobs


# --------------------------------------------------------------------------
# Command-line interface
# --------------------------------------------------------------------------


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Construct the command-line interface.

    Returns
    -------
    argparse.ArgumentParser
        Parser accepting either a configuration file or an input/output pair,
        plus the optional processing switches.

    Notes
    -----
    No filesystem path is hard-coded anywhere in this module. Paths arrive
    either from ``--config`` or as positional arguments, so a run is fully
    described by its command line together with a version-controlled
    configuration file, and can be recorded verbatim in a lab notebook or a
    workflow manager.
    """
    parser = argparse.ArgumentParser(
        prog="Horns.py",
        description=(
            "Calculate terrain slope in DEGREES from an elevation GeoTIFF "
            "using Horn's (1981) finite-difference method over the complete "
            "3x3 cell neighbourhood. The output is a co-registered float32 "
            "GeoTIFF of slope angle; it is not elevation and it is not "
            "percent slope."
        ),
        epilog=(
            "Examples:\n"
            "  python scripts/Horns.py --config config/horn_config.yaml\n"
            "  python scripts/Horns.py --config config/horn_config.yaml "
            "--job noaa_lidar\n"
            "  python scripts/Horns.py input_dem.tif output_slope.tif\n"
            "  python scripts/Horns.py arcticdem.tif slope.tif "
            "--apply-scale-factor\n"
            "\n"
            "The configuration file is the preferred route for research use: "
            "it records which inputs produced which outputs under which "
            "options, in one version-controllable artefact.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_raster",
        nargs="?",
        help=(
            "Path to the input elevation raster (DEM) GeoTIFF. "
            "Omit when using --config."
        ),
    )
    parser.add_argument(
        "output_raster",
        nargs="?",
        help=(
            "Path for the output slope GeoTIFF (values in DEGREES). "
            "Omit when using --config."
        ),
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        help=(
            "YAML configuration file specifying input and output paths and "
            "per-dataset options, so that paths never need to be typed on the "
            "command line or edited into the code. Mutually exclusive with "
            "the positional arguments. See config/horn_config.yaml."
        ),
    )
    parser.add_argument(
        "--job",
        metavar="NAME",
        action="append",
        help=(
            "Run only the named job(s) from the configuration file. May be "
            "repeated. Default: run every job in the file."
        ),
    )
    parser.add_argument(
        "--z-factor",
        type=float,
        default=1.0,
        metavar="FACTOR",
        help=(
            "Multiplier converting the vertical (elevation) unit into the "
            "horizontal linear unit of the CRS. Use 1.0 (default) when both "
            "are metres. Use 3.2808333 for metre elevations on a US survey "
            "foot grid, or 0.3048 for foot elevations on a metre grid."
        ),
    )
    parser.add_argument(
        "--nodata",
        type=float,
        default=DEFAULT_OUTPUT_NODATA,
        metavar="VALUE",
        help=(
            "NoData value for the output raster. Must lie outside the "
            "attainable slope range [0, 90) degrees. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--allow-geographic-crs",
        action="store_true",
        help=(
            "Proceed even if the DEM uses a geographic (degree-based) CRS. "
            "NOT RECOMMENDED: the resulting values are not physically "
            "meaningful slope angles, because the horizontal separation is an "
            "angle rather than a distance. Reproject the DEM instead."
        ),
    )
    parser.add_argument(
        "--apply-scale-factor",
        action="store_true",
        help=(
            "Correct for map projection distance distortion by dividing the "
            "nominal cell size by the projection's point scale factor, so "
            "derivatives are taken with respect to true ground distance. "
            "Recommended for ArcticDEM (EPSG:3413), where the bias reaches "
            "about +4%% at 60 N. Negligible for UTM. Off by default to match "
            "the convention of mainstream GIS slope tools."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help=("Suppress informational messages; report warnings and "
              "errors only."),
    )
    return parser


def validate_output_target(output_path: str, nodata_value: float,
                           overwrite: bool) -> None:
    """
    Check that the output can be written and that the NoData value is safe.

    Parameters
    ----------
    output_path : str
        Destination path for the slope raster.
    nodata_value : float
        Proposed NoData sentinel.
    overwrite : bool
        Whether an existing file may be replaced.

    Raises
    ------
    ValueError
        The NoData value is not finite or falls inside the attainable slope
        range [0, 90], so it would be indistinguishable from a genuine slope
        value; or the output already exists and ``overwrite`` is False; or the
        destination directory does not exist.

    Notes
    -----
    This runs BEFORE the DEM is read, so that a misconfigured run fails in
    under a second rather than after a long computation whose result could not
    have been written anyway.
    """
    if not math.isfinite(nodata_value):
        raise ValueError(
            f"Output NoData value must be finite, got {nodata_value}."
        )
    if 0.0 <= nodata_value <= MAX_ATTAINABLE_SLOPE_DEGREES:
        raise ValueError(
            f"Output NoData value {nodata_value} lies inside the attainable "
            "range of slope in degrees, [0, 90). A NoData sentinel must be "
            "distinguishable from a real measurement; choose a negative "
            "value such as -9999."
        )
    if os.path.exists(output_path) and not overwrite:
        raise ValueError(
            f"Output file already exists: {output_path}\n"
            "Pass --overwrite, or set overwrite: true in the configuration "
            "file, to replace it."
        )
    parent = os.path.dirname(os.path.abspath(output_path))
    if not os.path.isdir(parent):
        raise ValueError(f"Output directory does not exist: {parent}")


def run_job(job: HornJob) -> None:
    """
    Execute one complete Horn slope-derivation task.

    Parameters
    ----------
    job : HornJob
        Fully resolved task, from either the command line or a configuration
        file. By the time a job reaches this function its paths are absolute
        and its options are typed, so the two entry points are
        indistinguishable here and behave identically.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError, ValueError, RuntimeError
        Propagated from the validation, read, compute or write stages so that
        the caller can report the failure and set an exit status.

    Method
    ------
    Five stages, each delegated to a documented function so that any of them
    can be reused or tested in isolation:

    1. validate the output target and the input raster
       (:func:`validate_output_target`, :func:`validate_input_raster`);
    2. read the elevation array, cell sizes, CRS, axis orientation and
       projection scale factor (:func:`read_elevation_grid`);
    3. compute Horn's derivatives, the gradient magnitude and the slope angle
       (:func:`calculate_horn_slope`);
    4. compute and log quality-control statistics (:func:`print_statistics`);
    5. write the georeferenced output with provenance metadata
       (:func:`save_slope_raster`).
    """
    LOGGER.info("=" * 70)
    LOGGER.info("Job           : %s", job.name)
    LOGGER.info("Method        : Horn (1981) 3x3 finite differences")
    LOGGER.info("Input DEM     : %s", job.input_raster)
    LOGGER.info("Output slope  : %s (degrees)", job.output_raster)

    # Stage 1: validate before doing any expensive work.
    validate_output_target(job.output_raster, job.output_nodata, job.overwrite)
    validate_input_raster(job.input_raster,
                          allow_geographic_crs=job.allow_geographic_crs)

    # Stage 2: read elevation and the spatial parameters of the grid.
    grid = read_elevation_grid(
        job.input_raster,
        z_factor=job.z_factor,
        apply_scale_factor_correction=job.apply_scale_factor_correction,
    )
    LOGGER.info(
        "Cell size used: dx = %.10g, dy = %.10g %s (nominal, map plane)",
        grid.cell_size_x, grid.cell_size_y, grid.linear_unit,
    )
    LOGGER.info("Scale factor  : %s", grid.scale_factor_summary)

    # Stage 3: the finite-difference computation itself.
    result = calculate_horn_slope(grid)

    # Stage 4: quality control.
    stats = print_statistics(grid, result)

    # Stage 5: write the georeferenced product.
    save_slope_raster(
        output_path=job.output_raster,
        result=result,
        grid=grid,
        stats=stats,
        input_path=job.input_raster,
        z_factor=job.z_factor,
        nodata_value=job.output_nodata,
    )


def main(argv: Optional[list] = None) -> int:
    """
    Parse the command line and run one or more Horn slope-derivation jobs.

    Parameters
    ----------
    argv : list of str, optional
        Argument vector. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit status: 0 if every job succeeded, 1 if any failed.

    Method
    ------
    Paths reach this function by one of two routes, never by being edited into
    the source:

    - ``--config``, naming a YAML file that may define several jobs, optionally
      narrowed with one or more ``--job`` selectors; or
    - two positional arguments, for a one-off run.

    When a configuration file defines multiple jobs they run in file order. A
    failure in one job is logged and does not abort the remainder, so a batch
    over several datasets still processes everything it can; the exit status
    reflects whether any job failed, and a closing summary names the failures.
    """
    args = build_argument_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)-8s %(message)s",
        stream=sys.stdout,
    )

    # --- Resolve the command line into a list of jobs --------------------
    try:
        if args.config:
            if args.input_raster or args.output_raster:
                raise ValueError(
                    "--config cannot be combined with positional input/output "
                    "paths. Put the paths in the configuration file, or drop "
                    "--config and pass both positionally."
                )
            jobs = load_config(args.config)
            LOGGER.info("Loaded %d job(s) from %s", len(jobs), args.config)

            if args.job:
                wanted = list(args.job)
                available = {j.name for j in jobs}
                missing = [n for n in wanted if n not in available]
                if missing:
                    raise ValueError(
                        f"No job named {', '.join(repr(n) for n in missing)} "
                        f"in {args.config}.\n"
                        f"  Available jobs: {', '.join(sorted(available))}"
                    )
                jobs = [j for j in jobs if j.name in set(wanted)]

            # Command-line flags override the file for every selected job, so
            # a one-off variation needs no edit to a version-controlled config.
            if args.overwrite or args.apply_scale_factor:
                jobs = [
                    HornJob(
                        **{
                            **job.__dict__,
                            "overwrite": job.overwrite or args.overwrite,
                            "apply_scale_factor_correction": (
                                job.apply_scale_factor_correction
                                or args.apply_scale_factor
                            ),
                        }
                    )
                    for job in jobs
                ]
        else:
            if args.job:
                raise ValueError(
                    "--job is only meaningful together with --config."
                )
            if not args.input_raster or not args.output_raster:
                raise ValueError(
                    "Provide both an input and an output path, or use "
                    "--config to read them from a YAML file.\n"
                    "  python scripts/Horns.py dem.tif slope.tif\n"
                    "  python scripts/Horns.py --config "
                    "config/horn_config.yaml"
                )
            jobs = [
                HornJob(
                    name=os.path.splitext(
                        os.path.basename(args.input_raster)
                    )[0],
                    input_raster=os.path.abspath(args.input_raster),
                    output_raster=os.path.abspath(args.output_raster),
                    z_factor=args.z_factor,
                    output_nodata=args.nodata,
                    overwrite=args.overwrite,
                    apply_scale_factor_correction=args.apply_scale_factor,
                    allow_geographic_crs=args.allow_geographic_crs,
                )
            ]
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        return 1

    # --- Run them --------------------------------------------------------
    failures = []
    for job in jobs:
        try:
            run_job(job)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            LOGGER.error("Job '%s' failed: %s", job.name, exc)
            failures.append(job.name)
        except Exception as exc:  # pragma: no cover - unexpected failure
            LOGGER.exception("Job '%s' failed unexpectedly: %s", job.name, exc)
            failures.append(job.name)

    if len(jobs) > 1:
        LOGGER.info("=" * 70)
        LOGGER.info(
            "Completed %d of %d job(s).", len(jobs) - len(failures), len(jobs)
        )
    if failures:
        LOGGER.error("Failed job(s): %s", ", ".join(failures))
        return 1

    LOGGER.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
