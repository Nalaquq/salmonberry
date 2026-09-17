### 3. Slope Derivation via Central Differences

Given the traditional-knowledge hypothesis that salmonberry distribution tracks elevation, slope was derived directly from the DEM as a candidate predictor. A digital elevation model (DEM) represents a continuous elevation surface, $Z = Z(x,y)$, as a regular grid of elevation values with cell size $(dx, dy)$ equal to the pixel resolution of the source data. Slope is derived from the _gradient_ of this surface — the vector of its first partial derivatives:

$$
\nabla Z = \left( \frac{\partial Z}{\partial x}, \frac{\partial Z}{\partial y} \right)
$$

Because elevation is known only at discrete grid points rather than continuously, these derivatives must be approximated numerically. We used the **central differences method**, which estimates the derivative at a given cell from its two neighboring cells on either side, in each direction:

$$
Z'(x) \approx \frac{Z(x+dx) - Z(x-dx)}{2dx}
$$

This estimator can be derived from a Taylor expansion of the elevation surface about the focal cell in both the $+dx$ and $-dx$ directions. Subtracting the two expansions cancels both the focal elevation value and the second-order curvature term, leaving an approximation whose leading error scales with $dx^2$. This makes the central-difference estimate _second-order accurate_: halving the cell size reduces the truncation error roughly fourfold, compared to the coarser, first-order accuracy of simple forward or backward differences.

In practice, this means each cell's derivative is computed from its immediate neighbors — one column apart in the x-direction, one row apart in the y-direction — which requires at least three cells (one focal, two flanking) per estimate. Consequently, the derivative cannot be computed for the outermost row and column of the raster. Forward or backward differences could approximate slope at these edge cells, but doing so introduces bias and additional noise relative to the central-difference interior. We instead assigned NoData to edge cells, making this limitation explicit and machine-readable rather than silently degrading accuracy (see Appendix for the full NoData rule). This reduces the valid output raster to $(rows - 2) \times (cols - 2)$ cells — a negligible loss for a large DEM, but a meaningful one for small tiles. Tiled processing therefore used overlapping buffers of at least one cell, computing differences on the buffered tile and discarding the buffer afterward.

Once the partial derivatives were estimated, gradient magnitude was calculated as:

$$
|\nabla Z| = \sqrt{\left(\frac{\partial Z}{\partial x}\right)^2 + \left(\frac{\partial Z}{\partial y}\right)^2}
$$

Physically, $\nabla Z$ points in the direction of steepest ascent, and its magnitude gives the rate of elevation change per unit horizontal distance traveled in that direction. Because elevation and horizontal distance share the same linear unit, $|\nabla Z|$ is dimensionless (rise per unit run).

Slope angle, $\theta$, in degrees, was then obtained from the arctangent of the gradient magnitude:

$$
\theta = \frac{\arctan(|\nabla Z|) \times 180}{\pi}
$$

Care was taken to avoid geometric projection errors in this calculation (see Appendix).

# Appendix

## 3.1 Coordinate reference system

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

## 3.2 Vertical units and the z-factor

The gradient is dimensionless only if the elevation unit and the horizontal
unit are the SAME unit. If they differ (most commonly: a DEM in metres on a
grid in US survey feet), the elevation array must be converted first. The
`--z-factor` option multiplies elevation by a constant before differencing:

    z_factor = (1 horizontal linear unit) expressed in vertical units, inverted
             = vertical_unit_in_metres / horizontal_unit_in_metres

    metres elevation on a metre grid          -> z_factor = 1.0
    metres elevation on US survey foot grid   -> z_factor = 1 / 0.3048006096
                                                          = 3.2808333
    feet elevation on a metre grid            -> z_factor = 0.3048

The script reports the CRS linear unit so that this choice can be made and
documented explicitly. It cannot detect the vertical unit, which is rarely
encoded in a GeoTIFF; z_factor is the user's responsibility and defaults to 1.0.

## 3.3 Map projection scale distortion

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

This module can correct for the distortion. When `apply_scale_factor_correction`
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

---

NoData cells carry no elevation observation. Their stored value is a _sentinel
value_ - a numerical placeholder (commonly -9999, -32768, or NaN) that occupies
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
