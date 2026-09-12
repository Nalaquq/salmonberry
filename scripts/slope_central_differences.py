r"""
slope_central_differences.py
==================

Terrain slope from a Digital Elevation Model (DEM) using the second-order
CENTRAL FINITE-DIFFERENCE approximation of the elevation gradient.

Purpose
-------
This module derives a *slope angle* raster (in degrees) from an *elevation*
raster (a DEM, in linear vertical units such as metres). The finite-difference
arithmetic is written out explicitly in NumPy rather than delegated to a
black-box GIS tool, so that every step of the numerical procedure is visible,
auditable, and reproducible by another researcher.

Conceptual chain implemented here
---------------------------------
    Elevation  Z(x, y)                        [vertical linear units, e.g. m]
        |
        |  central finite differences over the 4-connected neighbourhood
        v
    Partial derivatives  dZ/dx , dZ/dy        [dimensionless: m of rise / m of run]
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
    Slope raster                              [degrees, 0 <= theta < 90]

The output is *slope* (equivalently *slope angle*). It is not "steepness
elevation"; it is not elevation of any kind. Its units are degrees of arc.


1. Mathematical formulation
---------------------------
A DEM is a discrete sampling of a continuous elevation surface

    Z = Z(x, y)

on a regular grid of cells of size (dx, dy). Slope is defined from the
*gradient* of that surface, which is the vector of its first partial
derivatives:

    grad Z = ( dZ/dx , dZ/dy )

Because the surface is only known at discrete sample points, the continuous
derivatives must be approximated numerically. This module uses the central
difference.

1.1 Why the central difference approximates the continuous derivative
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Expand the elevation surface in a Taylor series about the focal cell centre
x, in the x direction, at the two flanking sample points x + dx and x - dx:

    Z(x + dx) = Z(x) + dx Z'(x) + (dx^2/2) Z''(x) + (dx^3/6) Z'''(x) + O(dx^4)
    Z(x - dx) = Z(x) - dx Z'(x) + (dx^2/2) Z''(x) - (dx^3/6) Z'''(x) + O(dx^4)

Subtracting the second from the first cancels Z(x) and, critically, cancels
the entire second-order term (dx^2/2) Z''(x) because it carries the same sign
in both expansions:

    Z(x + dx) - Z(x - dx) = 2 dx Z'(x) + (dx^3/3) Z'''(x) + O(dx^5)

Dividing by 2 dx and rearranging:

    Z'(x) = [Z(x + dx) - Z(x - dx)] / (2 dx)  -  (dx^2/6) Z'''(x) + O(dx^4)

The first term on the right is the central-difference estimator used here. The
remaining terms are the truncation error, whose leading order is proportional
to dx^2. The central difference is therefore *second-order accurate*: halving
the cell size reduces the truncation error by a factor of about four.

1.2 Why central rather than forward or backward differences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The one-sided estimators are obtained from a single Taylor expansion:

    forward:   Z'(x) ~ [Z(x + dx) - Z(x)] / dx      - (dx/2) Z''(x) + O(dx^2)
    backward:  Z'(x) ~ [Z(x) - Z(x - dx)] / dx      + (dx/2) Z''(x) + O(dx^2)

Their leading truncation error is proportional to dx^1, i.e. they are only
*first-order* accurate, and the error term retains Z''(x), the local surface
curvature. Three consequences matter for terrain analysis:

  (a) Accuracy. For the same cell size the central difference is one order more
      accurate. On a 10 m DEM this is the difference between an error term
      scaling with 10 and one scaling with 100/6.

  (b) Bias / directional asymmetry. A one-sided estimator evaluates the slope
      of the chord between the focal cell and *one* neighbour, so the estimate
      is effectively anchored half a cell away from the focal cell centre. On
      convex or concave terrain (Z'' != 0) this introduces a systematic,
      direction-dependent bias: a forward difference and a backward difference
      at the same cell disagree, and the resulting slope raster inherits a
      half-cell spatial shift. The central difference is symmetric about the
      focal cell and is unbiased with respect to curvature to leading order.

  (c) Noise. DEMs carry vertical error. A one-sided difference divides the
      difference of two noisy samples by dx; the central difference divides by
      2 dx, halving the amplification of uncorrelated vertical noise into the
      derivative estimate.

The cost of the central difference is that it requires a valid neighbour on
*both* sides of the focal cell, which is why raster edges and cells adjacent to
NoData cannot be evaluated (Sections 4 and 5 below).

1.3 Gradient magnitude
~~~~~~~~~~~~~~~~~~~~~~
    |grad Z| = sqrt( (dZ/dx)^2 + (dZ/dy)^2 )

Physically, grad Z is the vector that points in the direction of *steepest
ascent* of the elevation surface at that location, and its magnitude is the
rate of elevation change per unit horizontal distance travelled in that
direction. It is the maximum directional derivative over all azimuths: no
direction of travel across that cell gains elevation faster than |grad Z| per
unit of horizontal distance. (The direction itself is the *aspect*, which this
module does not compute.)

Because elevation and horizontal distance are expressed in the same linear
unit, |grad Z| is dimensionless: metres of rise per metre of run.

1.4 Slope angle
~~~~~~~~~~~~~~~
The gradient magnitude is the tangent of the angle between the terrain surface
and the horizontal plane. Inverting:

    theta_radians = arctan( |grad Z| )
    theta_degrees = arctan( |grad Z| ) * 180 / pi

Terminology, kept strictly distinct throughout this module:

    elevation            Z                       vertical units (e.g. m)
    elevation gradient   (dZ/dx, dZ/dy)          dimensionless vector
    gradient magnitude   |grad Z|                dimensionless scalar (= tan theta)
    slope angle          theta = arctan|grad Z|  degrees (this module's output)
    slope percent        |grad Z| * 100          percent (NOT degrees)

Slope percent and slope degrees are different quantities and are not
interchangeable. A 100 % slope is 45 degrees, not 90; a 45 degree slope is
100 %, not 50 %. This module writes DEGREES to the output raster. Percent slope
is reported in the summary statistics for reference only.

Because arctan maps [0, inf) onto [0, 90) degrees, slope in degrees is bounded
above by 90 and is never exactly 90 for a finite gradient, whereas percent
slope is unbounded.


2. Raster neighbourhood and array orientation
---------------------------------------------
The central difference at focal cell (i, j) uses the four edge-adjacent cells:

                          Z(i-1, j)
                              |
                              |
         Z(i, j-1) ------ Z(i, j) ------ Z(i, j+1)
                              |
                              |
                          Z(i+1, j)

    Z(i, j-1)  west  / left  neighbour  (column index one lower)
    Z(i, j+1)  east  / right neighbour  (column index one higher)
    Z(i-1, j)  north / up    neighbour  (row index one lower)
    Z(i+1, j)  south / down  neighbour  (row index one higher)

Note that the focal value Z(i, j) itself does not enter either derivative
formula. It is nevertheless required to be valid data here, because a slope
value reported at a NoData cell would describe terrain that the DEM does not
claim to observe (Section 4).

2.1 Row index direction versus geographic northing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This is the single most common sign error in raster derivative code.

In a NumPy array read from a raster, the first axis is the *row* index i, and
i increases DOWNWARD through the array: row 0 is the top row of the image.
For a conventional "north-up" GeoTIFF, the top of the image is the northern
edge, so northing y DECREASES as i increases. The affine geotransform records
this explicitly: its pixel-height term (transform.e, the `e` coefficient) is
NEGATIVE for a north-up raster.

Therefore two different derivatives can be written, and they are not the same:

    row-index derivative (array space, downward-positive):

        dZ/di ~ [ Z(i+1, j) - Z(i-1, j) ] / (2 dy)

    geographic derivative (map space, northward-positive y):

        dZ/dy ~ [ Z(i-1, j) - Z(i+1, j) ] / (2 dy)     (north-up raster)

They differ by a factor of -1. This module computes the row-index derivative
first (it is the direct array expression), then multiplies by an orientation
factor derived from the sign of transform.e to obtain the geographic dZ/dy:

        orientation = -1 for a north-up raster (transform.e < 0)
        orientation = +1 for a south-up raster (transform.e > 0)
        dZ/dy = orientation * dZ/di

The x direction needs no such correction under the standard convention that
easting increases with column index (transform.a > 0); the code validates this
assumption rather than assuming it silently.

2.2 Why the sign does not change the answer, and why it is fixed anyway
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The gradient magnitude squares both components:

    |grad Z| = sqrt( (dZ/dx)^2 + (dZ/dy)^2 )

and (-a)^2 = a^2, so flipping the sign of the y derivative leaves the slope
magnitude, and hence the slope angle, completely unchanged. Getting this sign
wrong would not corrupt this module's output.

The sign is nevertheless handled correctly here for three reasons: the
intermediate quantity dZ/dy is reported and documented as a geographic partial
derivative and must therefore be the geographic one; any downstream use of the
signed components (aspect, curvature, flow direction, hillshade) depends
critically on the sign; and a method description in a publication should state
a derivative that is actually the derivative it names.


3. Cell size and coordinate reference system
--------------------------------------------
3.1 Cell size
~~~~~~~~~~~~~
dx and dy are read from the raster's affine transform at runtime and are never
hard-coded. dx is the absolute value of the pixel width (transform.a) and dy
the absolute value of the pixel height (transform.e). Non-square cells are
supported: dx and dy enter their respective derivative formulas separately.

The true resolution must be used because the finite difference is a *rate*:
it divides an elevation difference by the horizontal distance over which that
difference is realised. The same 20 m elevation drop between two cells implies
a gradient of 1.0 across 10 m cells but 0.1 across 100 m cells, i.e. 45 degrees
versus 5.7 degrees. Substituting a wrong cell size rescales the gradient
linearly and therefore biases every slope value in the product.

With elevation in metres and cell size in metres, the derivative has units of

    metres of elevation / metre of horizontal distance

which is dimensionless, as a tangent must be.

3.2 Coordinate reference system
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The horizontal denominator of the finite difference must be a real, physical
horizontal distance in the same linear unit as the elevation numerator. This
module therefore requires a PROJECTED CRS and refuses a geographic one by
default.

In a geographic CRS (e.g. EPSG:4326) the coordinates are angles, and the cell
size read from the transform is in DEGREES. Forming

    (metres of elevation) / (degrees of longitude)

is not a slope; it is a physically meaningless quantity, and arctan of it is
not an angle of terrain inclination. Two further problems compound this:

  (a) One degree of longitude is not a fixed ground distance. It shrinks as
      cos(latitude), from about 111.3 km at the equator to about 55.8 km at
      60 degrees N and to zero at the poles. A single scalar dx cannot
      describe a whole geographic raster.

  (b) One degree of longitude and one degree of latitude subtend different
      ground distances everywhere except the equator, so a "square" cell in
      degrees is not square on the ground, and dx and dy would be
      inconsistently scaled relative to each other.

The recommended practice is to reproject the DEM to a projected CRS with linear
units (for example a UTM zone, a national grid, or a local equal-area or
conformal projection appropriate to the study area) before running this module,
choosing a projection whose distance distortion over the study area is small.

3.3 Vertical units and the z-factor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The gradient is dimensionless only if the elevation unit and the horizontal
unit are the SAME unit. If they differ (most commonly: a DEM in metres on a
grid in US survey feet), the elevation array must be converted first. The
``--z-factor`` option multiplies elevation by a constant before differencing:

    z_factor = (1 horizontal linear unit) expressed in vertical units, inverted
             = vertical_unit_in_metres / horizontal_unit_in_metres

    metres elevation on a metre grid          -> z_factor = 1.0
    metres elevation on US survey foot grid   -> z_factor = 1 / 0.3048006096
                                                          = 3.2808333
    feet elevation on a metre grid            -> z_factor = 0.3048

The script reports the CRS linear unit so that this choice can be made and
documented explicitly. It cannot detect the vertical unit, which is rarely
encoded in a GeoTIFF; z_factor is the user's responsibility and defaults to 1.0.

3.4 Map projection scale distortion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A projected CRS gives horizontal coordinates in linear units, but those units
are measured ON THE MAP PLANE, not on the ground. Every map projection distorts
distance, and the ratio between the two is the point scale factor

    k = (distance on the map plane) / (distance on the ellipsoid)

so that the true ground separation of two cells whose nominal (map) separation
is D is D / k. Because the finite difference divides an elevation difference by
a horizontal distance, using the nominal cell size where k != 1 biases every
derivative by exactly the factor k:

    dZ/dx_true = k * dZ/dx_nominal

This is a systematic, spatially varying bias, not random error, and it does not
average out. It survives into any statistic computed from the slope raster.

Whether it matters depends entirely on the projection:

  - UTM: k ranges from 0.9996 on the central meridian to about 1.0010 at the
    zone edge. The worst-case bias is about 0.1 %, i.e. under 0.05 degrees for
    a 30 degree slope. Usually negligible, but not always ignorable.

  - Polar stereographic, as used by ArcticDEM (EPSG:3413), has its standard
    parallel at 70 N, where k = 1 exactly. Away from it k departs from 1
    substantially:

        latitude   k          bias in the gradient
        75 N       0.98666    -1.33 %
        70 N       1.00000     0.00 %   (standard parallel)
        65 N       1.01750    +1.75 %
        60 N       1.03998    +4.00 %
        55 N       1.06716    +6.72 %

    At 59.7 N a nominal 2 m ArcticDEM cell spans only 1.9214 m on the ground,
    and an uncorrected slope of 15.00 degrees is really 15.58 degrees.

This module can correct for the distortion. When ``apply_scale_factor_correction``
is enabled, the point scale factor is evaluated per cell from the CRS using
pyproj and the derivatives are divided by the TRUE ground cell size rather than
the nominal one. The correction is exact for conformal projections, where k is
isotropic (identical in every direction at a point) so a single factor applies
to both axes. Polar stereographic, UTM, Transverse Mercator and Lambert
Conformal Conic are all conformal, which covers ArcticDEM and effectively all
national and state lidar grids. For a non-conformal projection (an equal-area
grid, for instance) the module warns, because the distortion is then
anisotropic and correcting it rigorously requires the full distortion tensor
rather than a scalar.

The correction is OFF by default, so that the default output matches the
convention used by mainstream GIS slope tools, which work in nominal grid units.
Whenever it is off and the detected distortion exceeds 0.1 %, the module emits a
warning stating the measured k and the resulting bias, so the choice is never
made silently.


4. NoData handling
------------------
NoData cells carry no elevation observation. Their stored value is a *sentinel
value* - a numerical placeholder (commonly -9999, -32768, or NaN) that occupies
the cell to mark it as unobserved and is not a measurement of the ground. The
term is used here in its numerical-computing sense throughout; it has no
connection to the Sentinel Earth-observation missions, and this module neither
reads nor assumes any particular satellite product.

Both target datasets use -9999: ArcticDEM declares it in the GeoTIFF header,
and NOAA lidar DEMs conventionally do as well, though the header should always
be checked rather than assumed (this module reads the declared value and never
hard-codes one).

If a sentinel were allowed into the arithmetic the result would be catastrophic
rather than merely imprecise: a single -9999 m neighbour beside a 100 m focal
cell on a 10 m grid yields dZ/dx = (100 - (-9999)) / 20 = 505, i.e. a slope of
89.9 degrees where the terrain may be flat. The error is not local either, since
it propagates to every cell that uses the sentinel as a neighbour.

The rule applied here is strict and conservative:

    A cell receives a valid slope value only if the focal cell AND all four
    required neighbours (west, east, north, south) are valid data. Otherwise
    the output cell is set to NoData.

This means a NoData region in the DEM is dilated by exactly one cell in the
slope product, along the four cardinal directions. That is the intended and
documented behaviour: it guarantees that every slope value written to the
output is computed from five genuine elevation observations and from the
central-difference formula alone, with no substitution, no gap filling, no
one-sided fallback, and no implicit extrapolation across a data boundary. It is
preferable for scientific use to produce a smaller, fully defensible valid area
than a complete raster containing silently fabricated values.

Validity is determined from, in combination: the GDAL band mask (which honours
the declared nodata value and any internal or alpha mask), an explicit
comparison against the declared nodata value, and a finiteness test that
excludes NaN and +/- infinity.


5. Boundary handling
--------------------
The central difference at cell (i, j) requires cells at i-1, i+1, j-1 and j+1.
On the first and last row and the first and last column of the raster, at least
one of those indices falls outside the array, so the standard central-difference
equation is undefined there.

The outermost one-cell ring of the output raster is therefore set to NoData.

The alternative, substituting a one-sided (forward or backward) difference on
the edges, is rejected here. It would silently mix two estimators of different
order of accuracy (O(dx^2) in the interior, O(dx) on the edge) and different
bias behaviour within a single product. A user could not then tell from the
raster which estimator produced any given value, and edge statistics would not
be comparable with interior statistics. Assigning NoData makes the limitation
explicit and machine-readable instead of hiding it.

Practical effect: the valid area of the slope raster is (rows - 2) x (cols - 2)
cells, less any cells excluded by the NoData rule of Section 4. For a large DEM
this loss is negligible in proportion; for small tiles it is not, which is why
tiled processing of a DEM should be done with overlapping buffers of at least
one cell, differencing the buffered tile and then discarding the buffer.


6. Numerical precision
----------------------
All arithmetic is performed in 64-bit floating point (numpy.float64),
regardless of the input storage type.

Elevation is frequently stored as a signed integer (Int16, Int32) to save
space. Differencing integers in integer arithmetic would be wrong in two ways:
the quotient would be truncated toward zero, quantising every derivative to
whole units and destroying all slope information on gentle terrain (a 1 m rise
over 20 m of run would evaluate to 0, i.e. perfectly flat); and a sentinel such
as -32768 can overflow a narrow integer type during subtraction. The array is
therefore cast to float64 before any arithmetic.

float64 is used rather than float32 for the intermediate computation because
the finite difference is a *subtraction of nearby numbers* followed by division
by a small number, the classic setting for catastrophic cancellation: on gentle
terrain the two flanking elevations may agree to several significant digits, so
the leading digits cancel and the result is carried entirely by the trailing
digits. float64 provides about 15-16 significant decimal digits against
float32's 7, leaving ample margin after cancellation for realistic elevation
magnitudes.

The output is written as float32, which resolves a slope angle in degrees to
far better than 1e-5 degrees, several orders of magnitude finer than the
accuracy of any DEM-derived slope estimate, while halving file size. The
precision-critical work has already been done in float64 by that point.


7. Worked numerical example (see also docstring of ``slope_from_neighbours``)
-----------------------------------------------------------------------------
Given a focal cell on a grid with dx = 10 m, dy = 10 m, elevation in metres:

    Z_left  = Z(i, j-1) = 100 m      (west)
    Z_right = Z(i, j+1) = 120 m      (east)
    Z_up    = Z(i-1, j) = 105 m      (north)
    Z_down  = Z(i+1, j) = 115 m      (south)

Step 1, x derivative:

    dZ/dx = [Z(i, j+1) - Z(i, j-1)] / (2 dx)
          = (120 - 100) / (2 * 10)
          = 20 / 20
          = 1.0                          (dimensionless, m/m)

Step 2, y derivative. Row-index (downward) form first:

    dZ/di = [Z(i+1, j) - Z(i-1, j)] / (2 dy)
          = (115 - 105) / (2 * 10)
          = 10 / 20
          = 0.5

    For a north-up raster the geographic (northward-positive) derivative is

    dZ/dy = -1 * dZ/di = -0.5

    Interpretation: elevation increases toward the south, so the derivative
    with respect to northing is negative. The magnitude 0.5 is what enters the
    gradient.

Step 3, gradient magnitude:

    |grad Z| = sqrt( (dZ/dx)^2 + (dZ/dy)^2 )
             = sqrt( (1.0)^2 + (-0.5)^2 )
             = sqrt( 1.0 + 0.25 )
             = sqrt( 1.25 )
             = 1.118033988749895

Step 4, slope angle in radians:

    theta = arctan(1.118033988749895)
          = 0.8410686705679302 rad

Step 5, slope angle in degrees:

    theta_deg = 0.8410686705679302 * (180 / pi)
              = 0.8410686705679302 * 57.29577951308232
              = 48.189685104221404 degrees
              ~ 48.19 degrees

For contrast, the percent slope of the same cell is

    slope_percent = |grad Z| * 100 = 111.80 %

111.80 % and 48.19 degrees describe the identical surface. They are different
parameterisations of it and must never be reported interchangeably.


8. Relationship to GIS slope tools
----------------------------------
See the module README / accompanying methods text. In brief: GIS slope tools
also estimate the local elevation gradient from neighbouring cells and convert
it with an arctangent, but specific implementations differ in the
finite-difference kernel used (e.g. a 3x3 distance-weighted kernel over all
eight neighbours versus the 4-neighbour central difference implemented here),
in edge treatment, in NoData treatment, and in projection handling. Values from
this module should therefore be expected to be close to, but not bit-identical
with, values from any particular GIS package, and no claim of exact equivalence
is made here.


9. Dataset-specific notes: NOAA lidar and ArcticDEM
----------------------------------------------------
This module is source-agnostic, but the two datasets it is being applied to
have properties that materially affect how the resulting slope must be
interpreted and reported. They are documented here because the differences
are methodological, not incidental, and belong in any comparison of the two.

9.1 NOAA lidar (Digital Coast / Office for Coastal Management)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Sensor           Airborne lidar (direct ranging to the ground)
    Surface type     Usually BARE-EARTH DTM, ground returns classified and
                     vegetation/structures removed
    Resolution       Commonly 1 m, sometimes 0.5 m or 3 m
    Horizontal CRS   Usually UTM or State Plane, both conformal; scale
                     distortion typically under 0.1 %
    Vertical datum   NAVD88 (orthometric), via a geoid model
    Vertical units   METRES OR US SURVEY FEET, depending on the product.
                     This must be verified per tile. A foot-vertical DEM on a
                     foot-horizontal grid needs z_factor = 1.0; a foot-vertical
                     DEM on a metre grid needs z_factor = 0.3048.
    Vertical accuracy  Typically 0.10-0.20 m RMSE on open, firm ground; worse
                     under dense vegetation and on steep slopes
    NoData            Conventionally -9999

    Implications for slope. Bare-earth classification is an interpretive step,
    not a measurement: in dense vegetation, ground returns are sparse and the
    DTM is interpolated across gaps, so slope there reflects the interpolator
    as much as the terrain. Water surfaces absorb the near-infrared pulse and
    return little, so coastal and riverine areas carry extensive voids, which
    this module propagates as NoData dilated by one cell. Lidar is the more
    reliable source for true ground slope where coverage is good.

9.2 ArcticDEM (Polar Geospatial Center)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Sensor           Stereo-photogrammetry from sub-metre optical imagery
                     (WorldView / GeoEye). NOT lidar.
    Surface type     DIGITAL SURFACE MODEL. It records the first reflective
                     surface: vegetation canopy, buildings and snow, NOT bare
                     earth. There is no ground-return classification.
    Resolution       2 m strips; 2 m, 10 m and 32 m mosaics
    Horizontal CRS   EPSG:3413, WGS 84 / NSIDC Sea Ice Polar Stereographic
                     North. Conformal, standard parallel 70 N.
                     SCALE DISTORTION IS SUBSTANTIAL AWAY FROM 70 N - see
                     Section 3.4. At 60 N the bias is about +4 %.
    Vertical datum   Ellipsoidal heights (WGS84) for the standard products
    Vertical units   Metres
    NoData           -9999

    Implications for slope. Three points matter:

    (a) DSM, not DTM. Over vegetated or built terrain, ArcticDEM slope is the
        slope of the canopy or roof surface, not of the ground beneath it.
        Canopy-top roughness generates slope values that have no terrain
        meaning. Comparison with a bare-earth lidar DTM is therefore not a
        like-for-like comparison of the same surface, and any discrepancy
        between the two is partly a real difference in what was measured.
        Comparisons are most defensible over unvegetated ground: tundra,
        bare sediment, exposed bedrock, gravel bars.

    (b) Photogrammetric blunders. Stereo correlation fails on low-texture
        surfaces (uniform snow, still water, deep shadow) and under cloud,
        producing spikes, pits, and smoothed or interpolated patches. These
        are often inconspicuous in elevation but conspicuous in the
        derivative, since differentiation amplifies short-wavelength error.
        Mosaics additionally blend strips of different acquisition dates and
        can carry seam-line steps, which appear in slope as linear artefacts.
        Screening the slope raster for implausible values along seams is
        advisable.

    (c) Ellipsoidal versus orthometric heights are NOT a problem for slope.
        The two differ by the geoid undulation N, so slope is biased only by
        the GRADIENT of N. That gradient is of order 1e-4 to 1e-5 (tens of
        centimetres per kilometre), i.e. below 0.01 degrees, which is
        negligible against every other error source discussed here. A DEM in
        ellipsoidal heights therefore needs no conversion before computing
        slope, even though it would need one before comparing absolute
        elevations with a NAVD88 product.

9.3 Comparing the two datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A difference between NOAA lidar slope and ArcticDEM slope at the same location
confounds at least four distinct effects, which should not be attributed to
terrain change without separating them:

    1. Surface type    bare earth (DTM) versus canopy/first surface (DSM)
    2. Spatial scale   e.g. 1 m versus 2 m grids resolve different terrain
                       (Section on limitations; slope is scale-dependent)
    3. Projection      UTM/State Plane versus polar stereographic, with
                       different scale-factor bias unless corrected (3.4)
    4. Epoch           different acquisition dates, and mosaics blend dates

To make a defensible comparison, resample to a common grid and resolution,
apply the scale-factor correction to both, restrict the analysis to
unvegetated ground, and state the acquisition epochs.


Dependencies
------------
rasterio (GeoTIFF I/O, CRS, affine transform, NoData, metadata)
numpy    (array arithmetic)
pyproj   (point scale factor; optional, only for the distortion correction)
PyYAML   (configuration file; optional, only for --config)

ArcPy is deliberately not used, so the module runs unchanged inside an ArcGIS
Pro conda environment, in a plain virtual environment, or on a headless server.

Usage
-----
Paths are never hard-coded. Supply them either on the command line::

    python slope_central_differences.py elevation.tif slope.tif
    python slope_central_differences.py arcticdem.tif slope.tif --apply-scale-factor

or, preferably for repeatable research runs, in a YAML configuration file that
can be version-controlled alongside the results::

    python slope_central_differences.py --config config/slope_config.yaml
    python slope_central_differences.py --config config/slope_config.yaml --job arcticdem

See ``config/slope_config.yaml`` for the documented template.

Author / provenance
-------------------
Written for reproducible research use: the numerical method is implemented
explicitly and documented at methods-section level so that it can be described,
audited, and reimplemented independently.
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

# pyproj is required only for the map-projection scale-factor correction
# (module docstring, Section 3.4). It is a hard dependency of rasterio's CRS
# handling in practice, but the import is guarded so that the core
# finite-difference path still runs if it is unavailable.
try:
    from pyproj import Proj, CRS as PyprojCRS
    _PYPROJ_AVAILABLE = True
except ImportError:  # pragma: no cover - environment guard
    _PYPROJ_AVAILABLE = False

# PyYAML is required only for --config. Guarded so that the positional-argument
# command line works without it.
try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:  # pragma: no cover - environment guard
    _YAML_AVAILABLE = False


# --------------------------------------------------------------------------
# Module-level constants
# --------------------------------------------------------------------------

#: NoData value written to the output slope raster. Slope in degrees is
#: mathematically confined to [0, 90), so any negative sentinel is
#: unambiguous and can never collide with a legitimate slope value.
DEFAULT_OUTPUT_NODATA: float = -9999.0

#: Storage type of the output raster. Computation is done in float64
#: (see module docstring, Section 6); float32 storage resolves degrees to
#: far below the accuracy of any DEM-derived slope estimate.
OUTPUT_DTYPE: str = "float32"

#: Minimum raster dimension. A central difference needs one cell on each side
#: of the focal cell in both axes, so a 3x3 grid is the smallest raster with
#: any evaluable interior cell (exactly one).
MIN_RASTER_DIMENSION: int = 3

#: Relative departure of the point scale factor from 1 above which the module
#: warns that uncorrected map-projection distortion is biasing the result.
#: 0.001 (0.1 %) is below the noise floor of any DEM-derived slope, but it is
#: exceeded by an order of magnitude by ArcticDEM's polar stereographic grid
#: away from its 70 N standard parallel.
SCALE_FACTOR_WARN_THRESHOLD: float = 1e-3

#: Maximum number of lattice points per axis used to sample the scale-factor
#: field before bilinear interpolation onto the full grid. The field is an
#: extremely smooth function of position, so a coarse lattice is more than
#: adequate; this bounds the cost of the pyproj call on large rasters.
SCALE_FACTOR_LATTICE_MAX: int = 256

#: Relative spread of the scale factor across a raster below which a single
#: scalar is used instead of a full per-cell field, saving an array allocation
#: the size of the DEM.
SCALE_FACTOR_UNIFORM_TOLERANCE: float = 1e-9

#: Plausibility bounds for elevation in metres, used ONLY to emit a soft
#: warning that the band may not be an elevation surface. Lowest exposed land
#: ~ -430 m (Dead Sea shore); highest ~ 8849 m (Everest). Bathymetric or
#: non-metre DEMs legitimately fall outside this range, hence a warning and
#: never an error.
PLAUSIBLE_ELEVATION_RANGE_M: Tuple[float, float] = (-500.0, 9000.0)

LOGGER = logging.getLogger("slope_central_differences")


# --------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RasterGrid:
    """
    Container for a validated elevation grid and the spatial parameters that
    the finite-difference computation depends on.

    Attributes
    ----------
    elevation : numpy.ndarray
        2-D float64 array of elevation values, shape (rows, cols). Cells that
        are NoData, NaN or infinite have been replaced by ``numpy.nan`` so
        that no sentinel magnitude can leak into the arithmetic.
    valid_mask : numpy.ndarray
        2-D boolean array, True where ``elevation`` holds a genuine finite
        elevation observation. This is the authoritative record of data
        validity; ``elevation`` alone cannot be trusted because NaN
        comparisons are always False.
    cell_size_x : float
        Horizontal cell size dx along the x (column / easting) axis, in the
        CRS's linear units. Strictly positive.
    cell_size_y : float
        Horizontal cell size dy along the y (row / northing) axis, in the
        CRS's linear units. Strictly positive. Independent of ``cell_size_x``
        so that non-square cells are handled correctly.
    y_axis_orientation : int
        +1 or -1. The factor converting the row-index (downward-positive)
        derivative dZ/di into the geographic (northward-positive) derivative
        dZ/dy. It is -1 for the usual north-up raster, where increasing row
        index moves south. See module docstring, Section 2.1.
    transform : affine.Affine
        The raster's affine geotransform, carried through unmodified to the
        output so that the slope raster is co-registered with the input.
    crs : rasterio.crs.CRS
        The raster's coordinate reference system, carried through unmodified.
    linear_unit : str
        Human-readable name of the CRS's horizontal linear unit, used in
        reporting and in the output metadata.
    profile : dict
        The input raster's rasterio profile, used as the template for the
        output profile.
    scale_factor_x, scale_factor_y : float or numpy.ndarray
        Point scale factor k of the map projection along each axis (module
        docstring, Section 3.4). Either a scalar, where k is effectively
        uniform over the raster, or a 2-D float64 field. The nominal cell
        size is divided by k to recover the true ground cell size, which is
        algebraically identical to multiplying the derivative by k. Both are
        exactly 1.0 when the correction is disabled, in which case the
        arithmetic is unchanged.
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
    profile: dict
    scale_factor_x: object = 1.0
    scale_factor_y: object = 1.0
    scale_factor_applied: bool = False
    scale_factor_summary: str = "not applied (nominal grid units)"


@dataclass(frozen=True)
class SlopeResult:
    """
    Container for the computed slope raster and the intermediate gradient
    fields, retained so that callers can inspect or export the derivative
    components without recomputing them.

    Attributes
    ----------
    slope_degrees : numpy.ndarray
        2-D float64 array of slope angle in degrees, range [0, 90).
        ``numpy.nan`` marks cells that could not be evaluated (raster edge,
        or NoData in the 5-cell stencil).
    dz_dx : numpy.ndarray
        2-D float64 array of the partial derivative of elevation with respect
        to easting. Dimensionless. NaN where not evaluable.
    dz_dy : numpy.ndarray
        2-D float64 array of the partial derivative of elevation with respect
        to northing (geographic sign convention applied). Dimensionless.
        NaN where not evaluable.
    gradient_magnitude : numpy.ndarray
        2-D float64 array of |grad Z|, dimensionless, equal to tan(slope).
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


def slope_from_neighbours(
    z_left: float,
    z_right: float,
    z_up: float,
    z_down: float,
    cell_size_x: float,
    cell_size_y: float,
    y_axis_orientation: int = -1,
    scale_factor_x: float = 1.0,
    scale_factor_y: float = 1.0,
) -> Tuple[float, float, float, float]:
    """
    Scalar reference implementation of the central-difference slope for one cell.

    This function is the single-cell statement of exactly the same arithmetic
    that :func:`slope_central_differences_finite_difference` applies to the whole array
    with vectorised NumPy slicing. It exists so that the method can be read,
    reasoned about, and unit-tested one cell at a time, and so that the
    vectorised implementation can be verified against an unambiguous
    reference. It is not used in the production path over the raster, which
    is vectorised for performance.

    Parameters
    ----------
    z_left : float
        Elevation of the west neighbour, Z(i, j-1), in vertical units.
    z_right : float
        Elevation of the east neighbour, Z(i, j+1), in vertical units.
    z_up : float
        Elevation of the north neighbour, Z(i-1, j), in vertical units.
        "Up" means one row lower in array index, which is north on a
        north-up raster.
    z_down : float
        Elevation of the south neighbour, Z(i+1, j), in vertical units.
    cell_size_x : float
        dx, horizontal cell size along the column axis, in the same linear
        unit as the elevations.
    cell_size_y : float
        dy, horizontal cell size along the row axis, in the same linear unit
        as the elevations.
    y_axis_orientation : int, optional
        -1 for a north-up raster (default), +1 for a south-up raster. Converts
        the row-index derivative into the geographic derivative. Affects the
        sign of the returned ``dz_dy`` only, never the returned slope.
    scale_factor_x, scale_factor_y : float, optional
        Point scale factor k of the map projection along each axis, converting
        the nominal map-plane cell size into the true ground cell size
        (module docstring, Section 3.4). Default 1.0, i.e. no correction,
        reproducing the nominal-grid-unit convention of GIS slope tools.

    Returns
    -------
    tuple of (float, float, float, float)
        ``(dz_dx, dz_dy, gradient_magnitude, slope_degrees)``:

        - ``dz_dx``: partial derivative with respect to easting, dimensionless
        - ``dz_dy``: partial derivative with respect to northing, dimensionless
        - ``gradient_magnitude``: |grad Z|, dimensionless, equals tan(slope)
        - ``slope_degrees``: slope angle in degrees, in [0, 90)

    Method
    ------
    Second-order central finite differences over the 4-connected neighbourhood,
    followed by the Euclidean norm of the gradient vector and an arctangent.

    Mathematical formulation
    ------------------------
    ::

        dZ/dx = [ Z(i, j+1) - Z(i, j-1) ] / (2 dx)
        dZ/di = [ Z(i+1, j) - Z(i-1, j) ] / (2 dy)
        dZ/dy = orientation * dZ/di

        |grad Z| = sqrt( (dZ/dx)^2 + (dZ/dy)^2 )

        slope_degrees = arctan(|grad Z|) * 180 / pi

    Note that the focal elevation Z(i, j) does not appear: the central
    difference is formed entirely from the flanking neighbours.

    Assumptions
    -----------
    - All four neighbours are valid elevation observations (the caller is
      responsible for the NoData test; this function does not check).
    - Elevation and cell size share the same linear unit, so that the
      derivatives are dimensionless.
    - The grid is axis-aligned and regularly spaced.

    Notes
    -----
    Worked example, dx = dy = 10 m, elevations in metres::

        z_left = 100, z_right = 120, z_up = 105, z_down = 115

        dZ/dx    = (120 - 100) / (2 * 10) = 20 / 20 = 1.0
        dZ/di    = (115 - 105) / (2 * 10) = 10 / 20 = 0.5
        dZ/dy    = -1 * 0.5                         = -0.5
        |grad Z| = sqrt(1.0^2 + (-0.5)^2) = sqrt(1.25) = 1.118033988749895
        slope    = arctan(1.118033988749895) = 0.8410686705679302 rad
                 = 48.189685104221404 degrees

    The equivalent percent slope is |grad Z| * 100 = 111.80 %, which is the
    same terrain expressed differently, not a different steepness.

    Examples
    --------
    >>> dzdx, dzdy, grad, deg = slope_from_neighbours(100, 120, 105, 115, 10, 10)
    >>> round(dzdx, 10), round(dzdy, 10)
    (1.0, -0.5)
    >>> round(grad, 12)
    1.11803398875
    >>> round(deg, 6)
    48.189685
    """
    # Partial derivative with respect to easting. The numerator is the
    # elevation difference between the east and west neighbours; the
    # denominator 2*dx is the horizontal distance separating them, which is
    # two cell widths, not one.
    # The scale factor converts the nominal map-plane cell size into true
    # ground distance: dividing dx by k is identical to multiplying the
    # quotient by k. k = 1.0 leaves the arithmetic untouched.
    dz_dx = (z_right - z_left) / (2.0 * cell_size_x) * scale_factor_x

    # Partial derivative with respect to the row index, i.e. in the downward
    # (increasing-row) direction of the array.
    dz_di = (z_down - z_up) / (2.0 * cell_size_y) * scale_factor_y

    # Convert the array-space derivative into the geographic (northward
    # positive) derivative. For a north-up raster this negates it, because
    # moving down a row moves south.
    dz_dy = y_axis_orientation * dz_di

    # Magnitude of the gradient vector: the maximum rate of elevation change
    # per unit horizontal distance at this cell, in any direction.
    gradient_magnitude = math.sqrt(dz_dx * dz_dx + dz_dy * dz_dy)

    # The gradient magnitude is the tangent of the terrain inclination angle;
    # invert it to recover the angle, then express it in degrees.
    slope_degrees = math.degrees(math.atan(gradient_magnitude))

    return dz_dx, dz_dy, gradient_magnitude, slope_degrees


def gradient_magnitude_to_percent(gradient_magnitude: np.ndarray) -> np.ndarray:
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
    vertical face is infinite percent) while slope in degrees is bounded by 90.
    They coincide numerically nowhere except at zero. Reference values::

        |grad Z| = 0.0   ->    0 %   ->   0.00 degrees
        |grad Z| = 0.5   ->   50 %   ->  26.57 degrees
        |grad Z| = 1.0   ->  100 %   ->  45.00 degrees
        |grad Z| = 2.0   ->  200 %   ->  63.43 degrees

    Notes
    -----
    This module writes DEGREES to the output raster. This helper exists to
    report percent slope in the summary statistics, so that the distinction
    between the two parameterisations is visible in the run log and cannot be
    conflated by a later reader.
    """
    return gradient_magnitude * 100.0


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def _bilinear_expand(
    lattice: np.ndarray,
    lattice_rows: np.ndarray,
    lattice_cols: np.ndarray,
    rows: int,
    cols: int,
) -> np.ndarray:
    """
    Bilinearly interpolate a coarse lattice of values onto the full raster grid.

    Parameters
    ----------
    lattice : numpy.ndarray
        2-D array of values sampled at the lattice positions.
    lattice_rows, lattice_cols : numpy.ndarray
        1-D arrays of the fractional row and column indices at which
        ``lattice`` was sampled, in ascending order.
    rows, cols : int
        Dimensions of the target full-resolution grid.

    Returns
    -------
    numpy.ndarray
        2-D float64 array of shape ``(rows, cols)``.

    Method
    ------
    Standard separable bilinear interpolation, fully vectorised. Each target
    row and column index is mapped to a fractional position within the lattice
    with ``numpy.interp``, the four surrounding lattice values are gathered,
    and they are combined with the usual product weights.

    Notes
    -----
    This is used only for the map-projection scale factor, which is an
    analytic, monotone and extremely smooth function of position: over a
    single lattice interval it is very nearly linear, so the interpolation
    error is orders of magnitude below the 0.1 % threshold at which the
    distortion itself becomes worth correcting. Sampling the lattice rather
    than calling pyproj once per cell keeps the cost negligible on large DEMs.
    """
    target_rows = np.arange(rows, dtype=np.float64)
    target_cols = np.arange(cols, dtype=np.float64)

    # Fractional position of each target index within the lattice.
    fr = np.interp(target_rows, lattice_rows,
                   np.arange(lattice_rows.size, dtype=np.float64))
    fc = np.interp(target_cols, lattice_cols,
                   np.arange(lattice_cols.size, dtype=np.float64))

    r0 = np.clip(np.floor(fr).astype(np.intp), 0, lattice_rows.size - 1)
    c0 = np.clip(np.floor(fc).astype(np.intp), 0, lattice_cols.size - 1)
    r1 = np.clip(r0 + 1, 0, lattice_rows.size - 1)
    c1 = np.clip(c0 + 1, 0, lattice_cols.size - 1)

    wr = (fr - r0)[:, None]
    wc = (fc - c0)[None, :]

    top = lattice[np.ix_(r0, c0)] * (1.0 - wc) + lattice[np.ix_(r0, c1)] * wc
    bottom = lattice[np.ix_(r1, c0)] * (1.0 - wc) + lattice[np.ix_(r1, c1)] * wc
    return top * (1.0 - wr) + bottom * wr


def compute_scale_factor_field(
    crs: "CRS",
    transform: object,
    shape: Tuple[int, int],
) -> Tuple[object, object, str, float, float]:
    """
    Evaluate the map projection's point scale factor over the raster.

    Parameters
    ----------
    crs : rasterio.crs.CRS
        The raster's projected coordinate reference system.
    transform : affine.Affine
        The raster's affine geotransform, assumed axis-aligned (enforced by
        :func:`validate_input`).
    shape : tuple of (int, int)
        ``(rows, cols)`` of the raster.

    Returns
    -------
    tuple of (scale_x, scale_y, summary, max_departure, mean_k)
        ``scale_x`` and ``scale_y`` are the point scale factor along each grid
        axis, returned as a scalar float where the factor is effectively
        uniform over the raster and as a 2-D float64 field otherwise.
        ``summary`` is a human-readable description for logs and metadata.
        ``max_departure`` is the largest value of ``|k - 1|`` found, used to
        decide whether an uncorrected run deserves a warning. ``mean_k`` is the
        mean scale factor, whose position relative to 1 gives the DIRECTION of
        the bias: k > 1 means map distances exceed ground distances, so
        uncorrected gradients and slope angles are understated.

    Raises
    ------
    RuntimeError
        pyproj is unavailable, or the CRS cannot be interrogated for scale
        factors.

    Method
    ------
    The point scale factor k is the ratio of a differential distance on the
    map plane to the corresponding distance on the ellipsoid::

        k = (distance on the map) / (distance on the ground)

    so the true ground size of a cell whose nominal map size is D is D / k.
    pyproj's ``Proj.get_factors`` returns the meridional and parallel scale
    factors at a point, computed analytically from the projection definition.

    The factor is evaluated on a lattice of at most
    ``SCALE_FACTOR_LATTICE_MAX`` points per axis, placed at cell centres, and
    bilinearly interpolated onto the full grid (see
    :func:`_bilinear_expand`). Where the spread across the raster is below
    ``SCALE_FACTOR_UNIFORM_TOLERANCE`` a single scalar is returned instead,
    avoiding a full-size array allocation.

    Conformality
    ------------
    For a CONFORMAL projection the meridional and parallel scale factors are
    equal at every point, so k is isotropic: it is the same in every direction,
    and a single scalar correctly rescales both grid axes regardless of how
    they are oriented relative to the meridian. This is the rigorous case, and
    it covers every projection relevant here - polar stereographic (ArcticDEM,
    EPSG:3413), Transverse Mercator / UTM, and Lambert Conformal Conic (most
    State Plane zones).

    For a NON-CONFORMAL projection the two differ, distortion is anisotropic,
    and a rigorous correction requires the full distortion tensor together
    with the local bearing of each grid axis. The function warns in that case
    and falls back to applying the parallel scale to the x axis and the
    meridional scale to the y axis, which is correct only where the grid axes
    happen to align with the parallel and meridian directions. The result is
    then approximate and is labelled as such in the output metadata.

    Assumptions
    -----------
    - The geotransform is axis-aligned (no rotation or shear).
    - The CRS is projected; a geographic CRS has no meaningful linear scale
      factor and is rejected earlier by :func:`validate_input`.

    Notes
    -----
    The correction enters the finite difference as a division of the cell size
    by k, which is algebraically identical to multiplying the derivative by k::

        dZ/dx_true = [Z(i,j+1) - Z(i,j-1)] / (2 * dx / k) = k * dZ/dx_nominal

    Because k is a smooth, slowly varying field, this introduces no additional
    numerical error of consequence.
    """
    if not _PYPROJ_AVAILABLE:
        raise RuntimeError(
            "The scale-factor correction requires 'pyproj', which is not "
            "installed. Install it with 'pip install pyproj', or disable the "
            "correction (apply_scale_factor_correction: false)."
        )

    rows, cols = shape

    # Lattice of sample positions in array index space, always including the
    # first and last row/column so the field is interpolated, never
    # extrapolated.
    n_lat_rows = int(min(rows, SCALE_FACTOR_LATTICE_MAX))
    n_lat_cols = int(min(cols, SCALE_FACTOR_LATTICE_MAX))
    lattice_rows = np.linspace(0.0, rows - 1.0, n_lat_rows)
    lattice_cols = np.linspace(0.0, cols - 1.0, n_lat_cols)

    col_grid, row_grid = np.meshgrid(lattice_cols, lattice_rows)

    # Cell-centre map coordinates. The geotransform is known to be
    # axis-aligned at this point, so the rotation terms are omitted.
    x = transform.c + (col_grid + 0.5) * transform.a
    y = transform.f + (row_grid + 0.5) * transform.e

    try:
        projection = Proj(PyprojCRS.from_wkt(crs.to_wkt()))
        lon, lat = projection(x, y, inverse=True)
        factors = projection.get_factors(lon, lat, radians=False)
        meridional = np.asarray(factors.meridional_scale, dtype=np.float64)
        parallel = np.asarray(factors.parallel_scale, dtype=np.float64)
    except Exception as exc:
        raise RuntimeError(
            "Could not evaluate the map projection's scale factor for CRS "
            f"{crs.to_string()}: {exc}\n"
            "Disable the correction (apply_scale_factor_correction: false) to "
            "compute slope in nominal grid units instead."
        ) from exc

    if not (np.all(np.isfinite(meridional)) and np.all(np.isfinite(parallel))):
        raise RuntimeError(
            "The map projection's scale factor is not finite over part of the "
            f"raster extent for CRS {crs.to_string()}. The raster may extend "
            "beyond the projection's valid domain."
        )

    # Conformality test: for a conformal projection the meridional and
    # parallel scale factors coincide everywhere.
    asymmetry = float(np.max(np.abs(meridional - parallel) / meridional))
    conformal = asymmetry < 1e-6

    if conformal:
        # Isotropic: one factor serves both axes, exactly.
        k_x = parallel
        k_y = parallel
        conformality_note = "conformal projection, isotropic scale factor"
    else:
        LOGGER.warning(
            "CRS %s is NOT conformal (meridional and parallel scale factors "
            "differ by up to %.3g). The scale-factor correction assumes the "
            "grid x axis follows the parallel and the y axis the meridian, "
            "which is exact only where those directions align with the grid. "
            "Treat the corrected slope as approximate, or reproject to a "
            "conformal CRS.",
            crs.to_string(), asymmetry,
        )
        k_x = parallel
        k_y = meridional
        conformality_note = (
            f"NON-conformal projection, anisotropic scale factor "
            f"(asymmetry {asymmetry:.3g}); correction is approximate"
        )

    k_min = float(min(np.min(k_x), np.min(k_y)))
    k_max = float(max(np.max(k_x), np.max(k_y)))
    k_mean = float((np.mean(k_x) + np.mean(k_y)) / 2.0)
    max_departure = max(abs(k_min - 1.0), abs(k_max - 1.0))

    # Where the factor is effectively constant over the raster, collapse the
    # field to a scalar and skip the full-size allocation entirely.
    spread = (k_max - k_min) / k_mean if k_mean != 0.0 else 0.0
    if spread < SCALE_FACTOR_UNIFORM_TOLERANCE:
        summary = (
            f"k = {k_mean:.8f} (uniform; {conformality_note})"
        )
        return float(k_mean), float(k_mean), summary, max_departure, k_mean

    scale_x = _bilinear_expand(k_x, lattice_rows, lattice_cols, rows, cols)
    scale_y = _bilinear_expand(k_y, lattice_rows, lattice_cols, rows, cols)
    summary = (
        f"k varies from {k_min:.8f} to {k_max:.8f} across the raster "
        f"(mean {k_mean:.8f}; {conformality_note})"
    )
    return scale_x, scale_y, summary, max_departure, k_mean


def _describe_linear_unit(crs: "CRS") -> Tuple[str, Optional[float]]:
    """
    Report the horizontal linear unit of a projected CRS.

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
    correct ``--z-factor`` when the vertical unit differs from the horizontal
    unit. The lookup is defensive because some CRS definitions, particularly
    hand-written WKT or non-EPSG definitions, do not expose a unit factor.
    """
    unit_name = "unknown"
    unit_metres: Optional[float] = None
    try:
        if crs.linear_units:
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


def validate_input(
    input_path: str,
    allow_geographic_crs: bool = False,
) -> None:
    """
    Verify that a file exists and is a readable single-surface raster.

    Parameters
    ----------
    input_path : str
        Filesystem path to the candidate elevation GeoTIFF.
    allow_geographic_crs : bool, optional
        If True, a geographic CRS produces a loud warning instead of an error.
        Default False, which is the scientifically correct behaviour.

    Returns
    -------
    None
        The function raises on failure and returns nothing on success.

    Raises
    ------
    FileNotFoundError
        The path does not exist or is not a regular file.
    ValueError
        The file cannot be opened as a raster, declares no CRS, declares a
        geographic CRS while ``allow_geographic_crs`` is False, has a
        degenerate or rotated geotransform, is smaller than 3x3 cells, carries
        a non-numeric band type, or contains no valid finite values.

    Method
    ------
    The checks are ordered from cheapest and most fundamental to most
    expensive, so that an obviously wrong input fails immediately without a
    full band read:

    1. the path exists and is a file;
    2. the file opens as a raster and has at least one band;
    3. the band's storage type is real-numeric (not complex, not boolean);
    4. a CRS is declared;
    5. the CRS is projected, not geographic (see module docstring, 3.2);
    6. the geotransform is axis-aligned (no rotation / shear terms) and has
       strictly positive, finite pixel dimensions;
    7. the raster is at least 3x3 cells, the minimum for one interior cell;
    8. the band, once read, contains at least one valid finite value, and
       enough of them for the stencil to be satisfiable anywhere.

    Assumptions
    -----------
    The first band is the elevation surface. A multiband file is accepted with
    a warning; bands 2 and higher are ignored.

    Notes
    -----
    Check 6 rejects rotated ("sheared") geotransforms because the derivative
    formulas implemented here assume that array rows and columns are aligned
    with the CRS axes. On a rotated grid, moving one column does not move
    purely in x, so the separable central differences would mix the two
    directions and both partial derivatives would be wrong. Such a raster must
    be resampled to an axis-aligned grid before use.

    A soft plausibility warning is raised if elevation values fall far outside
    the range of terrestrial elevations in metres. This is a heuristic aid
    only; bathymetry and non-metre vertical units legitimately fall outside it,
    so it never blocks execution.
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

        # --- 4 & 5. CRS presence and projection type ---------------------
        if src.crs is None:
            raise ValueError(
                "Raster declares no coordinate reference system. Slope cannot "
                "be computed, because the physical meaning of the cell size "
                "is unknown. Assign or reproject to a projected CRS with "
                "linear units (e.g. a UTM zone) before running this script."
            )

        if src.crs.is_geographic:
            message = (
                "The raster uses a GEOGRAPHIC coordinate reference system "
                f"({src.crs.to_string()}), so its cell size is expressed in "
                "DEGREES of angle, not in linear distance.\n"
                "  Dividing an elevation difference in metres by a horizontal "
                "separation in degrees does not yield a slope: the quotient "
                "is not dimensionless and its arctangent is not a terrain "
                "angle.\n"
                "  One degree of longitude also varies with latitude (about "
                "111 km at the equator, about 56 km at 60 degrees latitude, "
                "zero at the poles), so no single cell size in degrees "
                "describes the raster.\n"
                "  Reproject the DEM to a projected CRS with linear units "
                "(for example the appropriate UTM zone or an equal-area "
                "projection for the study area) and re-run."
            )
            if not allow_geographic_crs:
                raise ValueError(message)
            LOGGER.warning(
                "%s\n  --allow-geographic-crs was supplied: continuing, but "
                "THE RESULTING VALUES ARE NOT PHYSICALLY MEANINGFUL SLOPES "
                "and must not be reported as slope angles.",
                message,
            )
        elif not src.crs.is_projected:
            LOGGER.warning(
                "CRS %s is neither clearly projected nor geographic "
                "(it may be a local engineering or unknown CRS). Verify that "
                "its horizontal units are linear distances before trusting "
                "the output.",
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
                f"(b={transform.b}, d={transform.d}). The separable central "
                "differences implemented here assume rows and columns are "
                "aligned with the CRS axes; on a rotated grid a one-column "
                "step is not a pure step in x. Resample the DEM to an "
                "axis-aligned grid before computing slope."
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
                f"Raster cell dimensions must be non-zero "
                f"(dx={cell_size_x}, dy={cell_size_y}). A zero cell size "
                "would make the finite-difference denominator zero."
            )

        # --- 7. Raster large enough for a central difference -------------
        if src.height < MIN_RASTER_DIMENSION or src.width < MIN_RASTER_DIMENSION:
            raise ValueError(
                f"Raster is {src.height} rows x {src.width} columns, which is "
                f"too small for a central finite difference. At least "
                f"{MIN_RASTER_DIMENSION}x{MIN_RASTER_DIMENSION} cells are "
                "required so that at least one cell has neighbours on both "
                "sides in both axes."
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
        if valid_count < MIN_RASTER_DIMENSION * MIN_RASTER_DIMENSION:
            LOGGER.warning(
                "Band 1 contains only %d valid cells. The 5-cell "
                "central-difference stencil may not be satisfiable anywhere, "
                "in which case the output will be entirely NoData.",
                valid_count,
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
                "unit of the CRS (see --z-factor).",
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
    Read an elevation raster and the spatial parameters the derivative needs.

    Parameters
    ----------
    input_path : str
        Path to a validated elevation GeoTIFF (call :func:`validate_input`
        first).
    z_factor : float, optional
        Multiplicative factor applied to elevation to express it in the same
        linear unit as the horizontal coordinates. Default 1.0, correct when
        elevation and the CRS share a unit (the usual metres-on-metres case).
        See module docstring, Section 3.3.
    apply_scale_factor_correction : bool, optional
        If True, evaluate the map projection's point scale factor k and divide
        the nominal cell size by it, so the derivative is taken with respect to
        true ground distance rather than map-plane distance. Default False,
        which matches the convention of mainstream GIS slope tools. Strongly
        recommended for ArcticDEM, whose polar stereographic grid carries a
        bias of roughly +4 % at 60 N. See module docstring, Section 3.4.

    Returns
    -------
    RasterGrid
        Elevation as float64 with invalid cells set to NaN, the boolean
        validity mask, dx, dy, the y-axis orientation factor, the scale-factor
        fields, and the CRS, transform and profile needed to write a
        co-registered output.

    Method
    ------
    1. Read band 1 and cast to float64 (module docstring, Section 6).
    2. Build the validity mask from three independent sources, combined with
       logical AND so that any one of them can exclude a cell:

       - ``read_masks``, GDAL's own band mask, which already accounts for the
         declared nodata value, an internal mask band, and an alpha band;
       - an explicit equality test against ``src.nodata``, which catches the
         case of a nodata value declared in metadata but not honoured by a
         mask;
       - ``numpy.isfinite``, which excludes NaN and +/- infinity regardless of
         whether they were declared as nodata. NaN in particular cannot be
         caught by equality, since ``NaN != NaN``.

    3. Overwrite every invalid cell with ``numpy.nan``. This is defensive
       redundancy: the validity mask alone is sufficient to exclude those
       cells from the output, but replacing sentinels with NaN guarantees
       that even a bug in the masking logic produces a visible NaN rather
       than a plausible-looking slope derived from a -9999 m "elevation".
    4. Apply the z-factor to the valid elevations.
    5. Derive dx and dy from the affine transform, and the y-axis orientation
       from the sign of the transform's pixel-height term.

    Assumptions
    -----------
    - Band 1 is the elevation surface.
    - The grid is axis-aligned and regularly spaced (enforced by
      :func:`validate_input`).
    - The whole band fits in memory. For DEMs too large for RAM, the same
      arithmetic should be applied over windows with a one-cell overlap so
      that the stencil is satisfiable across window boundaries.

    Notes
    -----
    ``transform.e`` is negative for the conventional north-up raster; its
    sign, not its magnitude, determines ``y_axis_orientation``. The magnitude
    is dy.
    """
    with rasterio.open(input_path) as src:
        raw_band = src.read(1)

        # Cast to float64 before any arithmetic. Integer DEMs would otherwise
        # truncate the quotient of the finite difference toward zero.
        elevation = raw_band.astype(np.float64, copy=True)

        # --- Validity mask, from three independent criteria ---------------
        valid_mask = src.read_masks(1) != 0
        if src.nodata is not None and np.isfinite(src.nodata):
            valid_mask &= raw_band != src.nodata
        valid_mask &= np.isfinite(elevation)

        # --- Neutralise every invalid cell --------------------------------
        # NaN propagates through arithmetic, so even if a masked cell were
        # mistakenly used its influence would surface as NaN rather than as a
        # silently wrong slope derived from a sentinel such as -9999.
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

        unit_name, _unit_metres = _describe_linear_unit(src.crs)

        # --- Map projection scale factor (module docstring, Section 3.4) ---
        # The cell size read above is a MAP-PLANE distance. The true ground
        # distance is that divided by the point scale factor k. Applying the
        # correction is algebraically identical to multiplying the derivative
        # by k, which is how it is used in the computation.
        scale_x: object = 1.0
        scale_y: object = 1.0
        scale_summary = "not applied (slope computed in nominal grid units)"
        applied = False

        if src.crs is not None and src.crs.is_projected:
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
            profile=src.profile.copy(),
            scale_factor_x=scale_x,
            scale_factor_y=scale_y,
            scale_factor_applied=applied,
            scale_factor_summary=scale_summary,
        )


# --------------------------------------------------------------------------
# Core computation
# --------------------------------------------------------------------------


def slope_central_differences_finite_difference(grid: RasterGrid) -> SlopeResult:
    """
    Compute slope in degrees by second-order central finite differences.

    This is the core scientific routine of the module. It evaluates the two
    first partial derivatives of the elevation surface at every interior cell
    whose full 5-cell stencil is valid data, combines them into a gradient
    magnitude, and converts that magnitude into a terrain inclination angle.

    Parameters
    ----------
    grid : RasterGrid
        Validated elevation grid with cell sizes, y-axis orientation and
        validity mask, as returned by :func:`read_elevation_grid`.

    Returns
    -------
    SlopeResult
        Slope in degrees, the two partial derivatives, the gradient magnitude,
        and the boolean mask of cells that received a defensible value. Cells
        not evaluated hold ``numpy.nan`` in all float arrays.

    Method
    ------
    For each interior cell (i, j), with 1 <= i <= rows-2 and
    1 <= j <= cols-2, the 4-connected neighbourhood is::

                          Z(i-1, j)                north / up
                              |
         Z(i, j-1) ------ Z(i, j) ------ Z(i, j+1)  west ... east
                              |
                          Z(i+1, j)                south / down

    The focal value Z(i, j) does not enter the derivative arithmetic; only
    the four flanking cells do.

    Mathematical formulation
    ------------------------
    ::

        dZ/dx (i,j) = [ Z(i, j+1) - Z(i, j-1) ] / (2 dx)

        dZ/di (i,j) = [ Z(i+1, j) - Z(i-1, j) ] / (2 dy)
        dZ/dy (i,j) = orientation * dZ/di (i,j)

        |grad Z|(i,j) = sqrt( (dZ/dx)^2 + (dZ/dy)^2 )

        slope_degrees(i,j) = arctan( |grad Z|(i,j) ) * 180 / pi

    where dx and dy are the horizontal cell dimensions in the CRS's linear
    units, and ``orientation`` is -1 for a north-up raster (module docstring,
    Section 2.1).

    Each derivative is second-order accurate, with truncation error
    O(dx^2) and O(dy^2) respectively; see module docstring, Section 1.1.

    Implementation
    --------------
    The formulas are applied with array slicing rather than a Python loop over
    cells. The slice ``Z[1:-1, 2:]`` is the array of east neighbours of every
    interior cell, ``Z[1:-1, :-2]`` the corresponding west neighbours, and so
    on; subtracting the two aligned slices evaluates the numerator for every
    interior cell at once. This is numerically identical to the per-cell
    arithmetic of :func:`slope_from_neighbours` (which serves as the scalar
    reference implementation) and is several orders of magnitude faster on a
    large DEM.

    NoData treatment
    ----------------
    A cell receives a slope value only if all five cells of its stencil - the
    focal cell and its four cardinal neighbours - are valid data. The focal
    cell is required even though its value is not used arithmetically, because
    a slope reported at a NoData cell would attribute terrain inclination to
    ground the DEM does not observe. Cells failing this test are set to NaN
    and excluded from the output mask. See module docstring, Section 4.

    Boundary treatment
    ------------------
    The outermost row and column on every side are set to NaN, because the
    central-difference stencil extends outside the array there. No one-sided
    fallback is substituted, so that a single, uniformly second-order
    estimator produced every value in the output. See module docstring,
    Section 5.

    Assumptions
    -----------
    - Elevation and cell size are in the same linear unit, so that the
      derivatives are dimensionless (enforced by the CRS check and the
      user-supplied z-factor).
    - The grid is regular and axis-aligned.
    - The DEM samples a single-valued surface Z(x, y): overhangs, caves and
      vertical faces cannot be represented, so slope is bounded below 90
      degrees by construction.

    Notes
    -----
    All intermediate arrays are float64. Non-finite results, which can only
    arise from an unmasked pathological input, are caught and demoted to NaN
    before return, so that no NaN or infinity can reach the output file.
    """
    elevation = grid.elevation
    valid = grid.valid_mask
    rows, cols = elevation.shape
    dx = grid.cell_size_x
    dy = grid.cell_size_y

    LOGGER.info(
        "Computing central finite differences on a %d x %d grid "
        "(dx = %.10g, dy = %.10g %s, y-axis orientation = %+d).",
        rows, cols, dx, dy, grid.linear_unit, grid.y_axis_orientation,
    )

    # Allocate output arrays pre-filled with NaN. Every cell that the method
    # cannot legitimately evaluate - the boundary ring and any cell with an
    # incomplete stencil - simply keeps its NaN, so "not evaluated" is the
    # default state and a value has to be earned.
    dz_dx = np.full((rows, cols), np.nan, dtype=np.float64)
    dz_dy = np.full((rows, cols), np.nan, dtype=np.float64)

    # ------------------------------------------------------------------
    # Named slices into the interior of the array.
    #
    # centre  : the focal cells (i, j)      for 1 <= i <= rows-2, 1 <= j <= cols-2
    # west    : their west  neighbours (i,   j-1)
    # east    : their east  neighbours (i,   j+1)
    # north   : their north neighbours (i-1, j)   -- one row UP in the array
    # south   : their south neighbours (i+1, j)   -- one row DOWN in the array
    #
    # All five slices have identical shape (rows-2, cols-2) and are aligned
    # cell-for-cell, so element k of one corresponds to element k of another.
    # ------------------------------------------------------------------
    centre = (slice(1, -1), slice(1, -1))
    west = (slice(1, -1), slice(None, -2))
    east = (slice(1, -1), slice(2, None))
    north = (slice(None, -2), slice(1, -1))
    south = (slice(2, None), slice(1, -1))

    # ------------------------------------------------------------------
    # Stencil completeness test.
    #
    # A slope value is computed only where the focal cell AND all four
    # cardinal neighbours hold genuine elevation observations. The logical
    # AND of the five aligned validity slices is exactly that condition.
    # ------------------------------------------------------------------
    stencil_valid = np.zeros((rows, cols), dtype=bool)
    stencil_valid[centre] = (
        valid[centre] & valid[west] & valid[east] & valid[north] & valid[south]
    )

    excluded_by_nodata = int(
        np.count_nonzero(valid[centre] & ~stencil_valid[centre])
    )
    if excluded_by_nodata:
        LOGGER.info(
            "%d interior cells hold valid elevation but have at least one "
            "NoData neighbour; they are assigned NoData in the slope output "
            "rather than being estimated from an incomplete stencil.",
            excluded_by_nodata,
        )

    # ------------------------------------------------------------------
    # Partial derivative of elevation with respect to easting.
    #
    #     dZ/dx = [ Z(i, j+1) - Z(i, j-1) ] / (2 dx)
    #
    # Numerator   : elevation difference between the east and west neighbours
    #               of the focal cell, in vertical units.
    # Denominator : 2 dx, the horizontal distance separating those two
    #               neighbours. It is TWO cell widths, because the neighbours
    #               are one cell either side of the focal cell.
    # Result      : dimensionless rise per unit run in the easting direction,
    #               positive where the terrain rises toward the east.
    # ------------------------------------------------------------------
    # The scale-factor correction (module docstring, Section 3.4) divides the
    # nominal cell size by the point scale factor k to obtain the true ground
    # cell size. Dividing the denominator by k is identical to multiplying the
    # quotient by k, which is the form used here:
    #
    #     dZ/dx = [Z(i,j+1) - Z(i,j-1)] / (2 * dx / k)
    #           = k * [Z(i,j+1) - Z(i,j-1)] / (2 * dx)
    #
    # k is exactly 1.0 when the correction is disabled, leaving the arithmetic
    # unchanged. Where k is a field it is sliced to the interior to align with
    # the derivative arrays.
    k_x = grid.scale_factor_x
    k_y = grid.scale_factor_y
    k_x_interior = k_x[centre] if isinstance(k_x, np.ndarray) else k_x
    k_y_interior = k_y[centre] if isinstance(k_y, np.ndarray) else k_y

    dz_dx[centre] = (
        (elevation[east] - elevation[west]) / (2.0 * dx) * k_x_interior
    )

    # ------------------------------------------------------------------
    # Partial derivative of elevation with respect to the ROW INDEX, i.e. in
    # the downward direction of the array:
    #
    #     dZ/di = [ Z(i+1, j) - Z(i-1, j) ] / (2 dy)
    #
    # This is positive where elevation increases with increasing row index,
    # which on a north-up raster means rising toward the SOUTH.
    # ------------------------------------------------------------------
    dz_di = (
        (elevation[south] - elevation[north]) / (2.0 * dy) * k_y_interior
    )

    # ------------------------------------------------------------------
    # Convert the row-index derivative into the geographic derivative with
    # respect to northing:
    #
    #     dZ/dy = orientation * dZ/di,   orientation = -1 for a north-up grid
    #
    # For a north-up raster, increasing row index moves south, so the two
    # derivatives are negatives of each other. The gradient magnitude below
    # squares this term, so the sign cannot change the slope; it is applied
    # because dz_dy is documented and returned as the derivative with respect
    # to northing and must therefore actually be that derivative, and because
    # any downstream signed use (aspect, curvature, flow routing) depends on
    # it. See module docstring, Section 2.2.
    # ------------------------------------------------------------------
    dz_dy[centre] = grid.y_axis_orientation * dz_di

    # ------------------------------------------------------------------
    # Gradient magnitude:
    #
    #     |grad Z| = sqrt( (dZ/dx)^2 + (dZ/dy)^2 )
    #
    # The Euclidean norm of the gradient vector. It is the maximum rate of
    # elevation change per unit horizontal distance at the cell, attained in
    # the direction of steepest ascent (whose azimuth is the aspect, not
    # computed here). Dimensionless, and equal to the tangent of the terrain
    # inclination angle.
    # ------------------------------------------------------------------
    gradient_magnitude = np.hypot(dz_dx, dz_dy)

    # ------------------------------------------------------------------
    # Slope angle:
    #
    #     theta = arctan(|grad Z|)                 [radians]
    #     theta_degrees = theta * 180 / pi         [degrees]
    #
    # arctan maps [0, inf) onto [0, 90) degrees, so the result is bounded:
    # a finite gradient can never produce exactly 90 degrees, consistent with
    # the fact that a single-valued surface Z(x, y) cannot represent a
    # vertical face.
    # ------------------------------------------------------------------
    slope_degrees = np.degrees(np.arctan(gradient_magnitude))

    # ------------------------------------------------------------------
    # Final sanitisation. The stencil test above should already exclude every
    # cell that could yield a non-finite result, but the output mask is
    # additionally intersected with a finiteness test so that no NaN or
    # infinity can possibly reach the written file, whatever the input.
    # ------------------------------------------------------------------
    output_valid_mask = stencil_valid & np.isfinite(slope_degrees)
    leaked = int(np.count_nonzero(stencil_valid & ~output_valid_mask))
    if leaked:
        LOGGER.warning(
            "%d cells passed the stencil validity test but produced a "
            "non-finite slope; they have been demoted to NoData. This "
            "indicates unusual input values and should be investigated.",
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


def report_statistics(grid: RasterGrid, result: SlopeResult) -> dict:
    """
    Compute and log quality-control statistics for the input and output.

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
        metadata: cell counts, and the minimum, maximum, mean, median and
        standard deviation of the valid slope values in degrees, plus the
        mean expressed as percent slope for comparison.

    Method
    ------
    Statistics are computed over the valid output cells only. Including
    NoData cells, whatever sentinel they carry, would corrupt every moment.
    The median is computed exactly with ``numpy.median`` rather than
    approximated; on a raster of any realistic size this is an O(n) selection
    and is inexpensive relative to the raster read.

    Notes
    -----
    The count of valid output cells is expected to be smaller than the count
    of valid input cells even on a gap-free DEM, because the boundary ring is
    always excluded (module docstring, Section 5). The difference between the
    two counts is decomposed in the log into the boundary contribution and the
    NoData-dilation contribution, so that an unexpectedly large loss is
    visible immediately.

    The mean slope in degrees is not the arctangent of the mean gradient, and
    the mean of a set of angles is not in general a meaningful summary of
    terrain; these statistics are reported for quality control (detecting an
    all-flat or all-vertical result, for instance), not as terrain parameters
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
        "boundary_cells_excluded": boundary_cells,
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
        "Raster dimensions            : %d rows x %d cols (%d cells)",
        rows, cols, total_cells,
    )
    LOGGER.info(
        "Valid elevation cells (input): %d (%.2f%% of raster)",
        valid_input, 100.0 * valid_input / total_cells,
    )
    LOGGER.info(
        "NoData cells (input)         : %d", stats["nodata_input_cells"],
    )
    LOGGER.info(
        "Valid slope cells (output)   : %d (%.2f%% of raster)",
        valid_output, 100.0 * valid_output / total_cells,
    )
    LOGGER.info(
        "NoData cells (output)        : %d", nodata_output,
    )
    LOGGER.info(
        "  of which boundary ring     : %d (central difference undefined)",
        boundary_cells,
    )
    LOGGER.info(
        "  of which NoData-adjacent   : %d (incomplete 5-cell stencil)",
        max(nodata_output - boundary_cells, 0),
    )

    if valid_output > 0:
        LOGGER.info("Slope minimum                : %.6f degrees",
                    stats["slope_min_degrees"])
        LOGGER.info("Slope maximum                : %.6f degrees",
                    stats["slope_max_degrees"])
        LOGGER.info("Slope mean                   : %.6f degrees",
                    stats["slope_mean_degrees"])
        LOGGER.info("Slope median                 : %.6f degrees",
                    stats["slope_median_degrees"])
        LOGGER.info("Slope standard deviation     : %.6f degrees",
                    stats["slope_stddev_degrees"])
        LOGGER.info("Mean gradient as percent slope: %.4f %% "
                    "(reported for contrast; the raster stores DEGREES)",
                    stats["slope_mean_percent"])
    else:
        LOGGER.warning(
            "No cell received a valid slope value. Every cell was either on "
            "the raster boundary or had an incomplete stencil. Check the "
            "extent of NoData in the input DEM."
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
    Write the slope raster as a georeferenced GeoTIFF co-registered with the input.

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
        Statistics from :func:`report_statistics`, a subset of which is
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
    The output profile is derived from the input profile so that spatial
    extent, CRS, affine transform and raster dimensions are preserved exactly
    and the slope raster overlays the DEM cell-for-cell. Only the properties
    that must change are overridden:

    - ``count`` is set to 1 (slope is a single scalar surface);
    - ``dtype`` is set to float32 (module docstring, Section 6);
    - ``nodata`` is set to the output sentinel, which is outside the
      mathematically attainable range [0, 90) of slope in degrees and so can
      never be confused with a real value;
    - LZW compression and tiling are enabled, which are lossless and change
      no pixel value.

    Immediately before writing, every invalid cell is replaced by the NoData
    sentinel and the array is asserted to be entirely finite, so that NaN and
    infinity cannot be written to the file.

    Notes
    -----
    Metadata tags record only facts that are true of this product: the
    derived quantity, the numerical method, the units, the stencil, the
    boundary and NoData rules, the cell size actually used, and the source
    file. No tag asserts equivalence with any GIS package's slope tool, and
    none records a quantity the file does not contain.
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
    # input profile; restate them explicitly so the guarantee is visible.
    profile.update(
        width=grid.profile["width"],
        height=grid.profile["height"],
        transform=grid.transform,
        crs=grid.crs,
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(output_array, 1)

        # Band-level description, shown by gdalinfo and by most GIS clients.
        dst.set_band_description(1, "Slope angle (degrees)")

        # Dataset-level provenance tags. Every entry is a verifiable statement
        # about how this specific file was produced.
        dst.update_tags(
            derived_product="Slope",
            input_variable="elevation",
            method="Central finite differences",
            finite_difference_method="central",
            finite_difference_order="2",
            finite_difference_stencil="4-connected (west, east, north, south)",
            slope_units="degrees",
            units="degrees",
            slope_value_range="[0, 90)",
            gradient_definition="sqrt((dZ/dx)^2 + (dZ/dy)^2)",
            slope_definition="arctan(gradient_magnitude) * 180 / pi",
            cell_size_x=f"{grid.cell_size_x:.10g}",
            cell_size_y=f"{grid.cell_size_y:.10g}",
            horizontal_linear_unit=grid.linear_unit,
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
                               "(central difference undefined at edges)",
            nodata_treatment="Cell assigned NoData unless focal cell and all "
                             "four cardinal neighbours are valid",
            computation_precision="float64 internally, float32 storage",
            source_dem=os.path.basename(input_path),
            valid_output_cells=str(stats.get("valid_output_cells", "")),
            nodata_output_cells=str(stats.get("nodata_output_cells", "")),
            software="slope_central_differences.py (rasterio + numpy)",
        )

    LOGGER.info("Slope raster written: %s", output_path)
    LOGGER.info(
        "  driver=GTiff  dtype=%s  nodata=%.1f  size=%d x %d  crs=%s",
        OUTPUT_DTYPE, nodata_value,
        grid.profile["height"], grid.profile["width"],
        grid.crs.to_string(),
    )


# --------------------------------------------------------------------------
# Command-line interface
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
class SlopeJob:
    """
    One fully resolved slope-derivation task.

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
        Vertical-to-horizontal unit conversion (module docstring, Section 3.3).
    output_nodata : float
        NoData sentinel for the output.
    overwrite : bool
        Whether an existing output may be replaced.
    apply_scale_factor_correction : bool
        Whether to correct for map projection distortion (Section 3.4).
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
        Path to the YAML configuration file.

    Returns
    -------
    list of SlopeJob
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
        raise ValueError(
            f"'jobs' must be a non-empty list in {config_path}."
        )

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
                    f"Job '{label}' in {config_path}: '{flag}' must be true or "
                    f"false, got {value!r}."
                )

        jobs.append(
            SlopeJob(
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


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Construct the command-line interface.

    Returns
    -------
    argparse.ArgumentParser
        Parser accepting the input DEM path, the output slope path, and the
        optional processing switches.

    Notes
    -----
    No filesystem path is hard-coded anywhere in this module; both the input
    and the output are required positional arguments, so a run is fully
    described by its command line and can be recorded verbatim in a lab
    notebook or a workflow manager.
    """
    parser = argparse.ArgumentParser(
        prog="slope_central_differences.py",
        description=(
            "Calculate terrain slope in degrees from an elevation GeoTIFF "
            "using second-order central finite differences."
        ),
        epilog=(
            "Examples:\n"
            "  python slope_central_differences.py lidar_dem.tif lidar_slope.tif\n"
            "  python slope_central_differences.py arcticdem.tif slope.tif "
            "--apply-scale-factor\n"
            "  python slope_central_differences.py --config config/slope_config.yaml\n"
            "  python slope_central_differences.py --config config/slope_config.yaml "
            "--job arcticdem\n"
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
            "Path for the output slope GeoTIFF (values in degrees). "
            "Omit when using --config."
        ),
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        help=(
            "YAML configuration file specifying input and output paths and "
            "per-dataset options, so that paths never need to be typed on the "
            "command line or edited into the code. Mutually exclusive with the "
            "positional arguments. See config/slope_config.yaml."
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
            "attainable slope range [0, 90). Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--allow-geographic-crs",
        action="store_true",
        help=(
            "Proceed even if the DEM uses a geographic (degree-based) CRS. "
            "NOT RECOMMENDED: the resulting values are not physically "
            "meaningful slope angles, because the horizontal separation is "
            "an angle rather than a distance. Reproject instead."
        ),
    )
    parser.add_argument(
        "--apply-scale-factor",
        action="store_true",
        help=(
            "Correct for map projection distance distortion by dividing the "
            "nominal cell size by the projection's point scale factor, so "
            "derivatives are taken with respect to true ground distance. "
            "STRONGLY RECOMMENDED for ArcticDEM (EPSG:3413), where the bias "
            "reaches about +4%% at 60 N. Negligible for UTM. Off by default "
            "to match the convention of mainstream GIS slope tools."
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
        help="Suppress informational messages; report warnings and errors only.",
    )
    return parser


def _validate_output_target(output_path: str, nodata_value: float,
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
        The NoData value falls inside the attainable slope range [0, 90], so
        it would be indistinguishable from a genuine slope value; or the
        output already exists and ``overwrite`` is False; or the destination
        directory does not exist.
    """
    if not math.isfinite(nodata_value):
        raise ValueError(
            f"Output NoData value must be finite, got {nodata_value}."
        )
    if 0.0 <= nodata_value <= 90.0:
        raise ValueError(
            f"Output NoData value {nodata_value} lies inside the attainable "
            "range of slope in degrees, [0, 90). A NoData sentinel must be "
            "distinguishable from a real measurement; choose a negative "
            "value such as -9999."
        )
    if os.path.exists(output_path) and not overwrite:
        raise ValueError(
            f"Output file already exists: {output_path}\n"
            "Pass --overwrite to replace it."
        )
    parent = os.path.dirname(os.path.abspath(output_path))
    if not os.path.isdir(parent):
        raise ValueError(f"Output directory does not exist: {parent}")


def run_job(job: SlopeJob) -> None:
    """
    Execute one complete slope-derivation task.

    Parameters
    ----------
    job : SlopeJob
        Fully resolved task, from either the command line or a configuration
        file. By the time a job reaches this function its paths are absolute
        and its options are typed, so the two entry points are indistinguishable
        here and behave identically.

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
       (:func:`_validate_output_target`, :func:`validate_input`);
    2. read the elevation array, cell sizes, CRS, axis orientation and
       projection scale factor (:func:`read_elevation_grid`);
    3. compute the derivatives, gradient magnitude and slope angle
       (:func:`slope_central_differences_finite_difference`);
    4. compute and log quality-control statistics (:func:`report_statistics`);
    5. write the georeferenced output with provenance metadata
       (:func:`save_slope_raster`).

    Notes
    -----
    Validation precedes computation so that a misconfigured run fails before
    any expensive work, and the output target is checked before the DEM is read
    so that a long computation is never discarded at the write step.
    """
    LOGGER.info("=" * 70)
    LOGGER.info("Job           : %s", job.name)
    LOGGER.info("Input DEM     : %s", job.input_raster)
    LOGGER.info("Output slope  : %s", job.output_raster)

    # Stage 1: validate before doing any expensive work.
    _validate_output_target(job.output_raster, job.output_nodata, job.overwrite)
    validate_input(job.input_raster,
                   allow_geographic_crs=job.allow_geographic_crs)

    # Stage 2: read elevation and the spatial parameters of the grid.
    grid = read_elevation_grid(
        job.input_raster,
        z_factor=job.z_factor,
        apply_scale_factor_correction=job.apply_scale_factor_correction,
    )
    LOGGER.info(
        "CRS           : %s (horizontal unit: %s)",
        grid.crs.to_string(), grid.linear_unit,
    )
    LOGGER.info(
        "Cell size     : dx = %.10g, dy = %.10g %s (nominal, map plane)",
        grid.cell_size_x, grid.cell_size_y, grid.linear_unit,
    )
    LOGGER.info("Scale factor  : %s", grid.scale_factor_summary)

    # Stage 3: the finite-difference computation itself.
    result = slope_central_differences_finite_difference(grid)

    # Stage 4: quality control.
    stats = report_statistics(grid, result)

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
    Parse the command line and run one or more slope-derivation jobs.

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

    - two positional arguments, for a one-off run; or
    - ``--config``, naming a YAML file that may define several jobs, optionally
      narrowed with one or more ``--job`` selectors.

    When a configuration file defines multiple jobs they run in file order. A
    failure in one job is logged and does not abort the remainder, so a batch
    over several datasets still processes everything it can; the exit status
    reflects whether any job failed, and a closing summary names the failures.

    Notes
    -----
    A configuration file is preferable for research use because it is a single
    version-controllable artefact recording exactly which inputs produced which
    outputs under which options - including whether the projection scale-factor
    correction was applied, which materially changes the values.
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
            LOGGER.info(
                "Loaded %d job(s) from %s", len(jobs), args.config
            )

            if args.job:
                wanted = list(args.job)
                available = {j.name for j in jobs}
                missing = [n for n in wanted if n not in available]
                if missing:
                    raise ValueError(
                        f"No job named {', '.join(repr(n) for n in missing)} "
                        f"in {args.config}.\n"
                        f"  Available jobs: "
                        f"{', '.join(sorted(available))}"
                    )
                jobs = [j for j in jobs if j.name in set(wanted)]

            # Command-line flags override the file for every selected job, so
            # a one-off variation needs no edit to a version-controlled config.
            if args.overwrite or args.apply_scale_factor:
                jobs = [
                    SlopeJob(
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
                raise ValueError("--job is only meaningful together with --config.")
            if not args.input_raster or not args.output_raster:
                raise ValueError(
                    "Provide both an input and an output path, or use "
                    "--config to read them from a YAML file.\n"
                    "  python slope_central_differences.py input.tif output.tif\n"
                    "  python slope_central_differences.py --config config/slope_config.yaml"
                )
            jobs = [
                SlopeJob(
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
