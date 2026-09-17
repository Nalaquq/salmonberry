### 3. Slope Derivation via Horn's Method

Given the traditional-knowledge hypothesis that salmonberry distribution tracks elevation, slope was derived directly from the DEM as a candidate predictor. A digital elevation model (DEM) represents a continuous elevation surface, $Z = Z(x,y)$, as a regular grid of elevation values with cell size $(dx, dy)$ equal to the pixel resolution of the source data. Slope is derived from the _gradient_ of this surface — the vector of its first partial derivatives:

$$
\nabla Z = \left( \frac{\partial Z}{\partial x}, \frac{\partial Z}{\partial y} \right)
$$

Because elevation is known only at discrete grid points rather than continuously, these derivatives must be approximated numerically. We used **Horn's (1981) method**, a finite-difference estimator that reads the complete $3 \times 3$ block of cells surrounding each focal cell. Labeling the nine cells in reading order, with $z_5$ the focal cell whose slope is being computed:

$$
\begin{matrix}
z_1 & z_2 & z_3 \\
z_4 & z_5 & z_6 \\
z_7 & z_8 & z_9
\end{matrix}
$$

Here $z_2$ and $z_8$ lie directly north and south of the focal cell, $z_4$ and $z_6$ directly west and east, and $z_1$, $z_3$, $z_7$ and $z_9$ are the four diagonal neighbours. In array terms, $z_1 \dots z_3$ occupy the row above the focal cell (one row lower in index, which is north on a north-up raster), $z_4 \dots z_6$ the focal row, and $z_7 \dots z_9$ the row below.

Horn's estimator forms a weighted difference between opposite sides of this window. In the $x$ (easting) direction it differences the east column against the west column:

$$
\frac{\partial Z}{\partial x} \approx \frac{(z_3 + 2z_6 + z_9) - (z_1 + 2z_4 + z_7)}{8\,dx}
$$

and in the row direction it differences the bottom row against the top row (south and north):

$$
\frac{\partial Z}{\partial y} \approx \frac{(z_7 + 2z_8 + z_9) - (z_1 + 2z_2 + z_3)}{8\,dy}
$$

The weights are the method's defining feature: the cells lying directly along the axis of differentiation ($z_4, z_6$ for $x$; $z_2, z_8$ for $y$) receive weight 2, while the four diagonal cells receive weight 1. The denominator $8\,dx$ follows from the weights, since each of the four weighted units on one side is compared with its counterpart $2\,dx$ away: $4 \times 2\,dx = 8\,dx$. Equivalently, and more transparently, Horn's estimate is the $1:2:1$ weighted mean of three ordinary central differences taken along three parallel lines through the window:

$$
\frac{\partial Z}{\partial x} \approx \frac{1 \cdot \frac{z_3 - z_1}{2dx} + 2 \cdot \frac{z_6 - z_4}{2dx} + 1 \cdot \frac{z_9 - z_7}{2dx}}{1 + 2 + 1}
$$

Each of the three bracketed terms is a slope estimate in its own right; Horn's rule averages them. Averaging three parallel estimates reduces the influence of vertical error in any single elevation sample — for independent cell errors of standard deviation $s$, the standard deviation of the derivative estimate falls from $0.707\,s/dx$ for the central difference to $0.433\,s/dx$ — but the same transverse averaging is indifferent to whether the variation it smooths is noise or real terrain, so narrow features such as gully walls, terrace risers and road cuts are represented less sharply and extreme values are attenuated. The two estimators agree exactly on a planar surface and diverge only where the surface is curved across the direction of differentiation. Neither is universally preferable; Horn's method was adopted here because its response to the interpolation noise of a high-resolution DEM is more conservative, and because it is the estimator implemented by the slope tools of mainstream GIS packages, which makes the product comparable with published terrain layers. The companion product derived by central differences (`docs/method_central_differences.md`) is retained for comparison, both slope rasters being computed from the same DEM with the same cell size, z-factor and scale-factor treatment so that the estimator is the only thing that differs between them.

Horn's estimate requires a complete $3 \times 3$ window, so the derivative cannot be computed for the outermost row and column of the raster. As with the central-difference product, we assigned NoData to these edge cells rather than substituting a one-sided estimate, which would introduce bias and additional noise relative to the interior (see Appendix for the full NoData rule). This reduces the valid output raster to $(rows - 2) \times (cols - 2)$ cells. Tiled processing therefore used overlapping buffers of at least one cell, computing the differences on the buffered tile and discarding the buffer afterward.

Once the partial derivatives were estimated, gradient magnitude was calculated as:

$$
|\nabla Z| = \sqrt{\left(\frac{\partial Z}{\partial x}\right)^2 + \left(\frac{\partial Z}{\partial y}\right)^2}
$$

Physically, $\nabla Z$ points in the direction of steepest ascent, and its magnitude gives the rate of elevation change per unit horizontal distance traveled in that direction. Because elevation and horizontal distance share the same linear unit, $|\nabla Z|$ is dimensionless (rise per unit run).

Slope angle, $\theta$, in degrees, was then obtained from the arctangent of the gradient magnitude:

$$
\theta = \frac{\arctan(|\nabla Z|) \times 180}{\pi}
$$

Care was taken to avoid geometric projection errors in this calculation (see Appendix). The resulting raster stores **slope angle in degrees**, mathematically confined to $0 \leq \theta < 90$; it is not elevation, and it is not percent slope ($|\nabla Z| \times 100$), which is an unbounded alternative parameterisation of the same gradient reported only in the run log.

# Appendix

## 3.1 Coordinate reference system

The horizontal denominators $8\,dx$ and $8\,dy$ must be real, physical
horizontal distances in the same linear unit as the elevation numerator. This
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

The module reports the input CRS, its type (projected or geographic), the x and
y resolution, the raster dimensions and the declared NoData value on every run,
so the basis of the calculation is recorded in the log rather than assumed.

## 3.2 Vertical units and the z-factor

The gradient is dimensionless only if the elevation unit and the horizontal
unit are the SAME unit. If they differ (most commonly: a DEM in metres on a
grid in US survey feet), the elevation array must be converted first. The
`z_factor` option multiplies elevation by a constant before differencing:

    z_factor = vertical_unit_in_metres / horizontal_unit_in_metres

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
average out. It affects Horn's method exactly as it affects the central
difference, since both divide by a horizontal distance.

Whether it matters depends entirely on the projection. For UTM, k ranges from
0.9996 on the central meridian to about 1.0010 at the zone edge, a worst-case
bias of about 0.1 %. For the polar stereographic grid used by ArcticDEM
(EPSG:3413), whose standard parallel is 70 N, k departs from 1 substantially
away from that parallel: about +1.75 % at 65 N and +4.00 % at 60 N, so that at
59.7 N a nominal 2 m cell spans only 1.9214 m on the ground and an uncorrected
slope of 15.00 degrees is really 15.58 degrees. The full table is given in
`docs/method_central_differences.md`, section 3.3.

When `apply_scale_factor_correction` is enabled, the point scale factor is
evaluated per cell from the CRS using pyproj and the derivatives are divided by
the TRUE ground cell size rather than the nominal one. The correction is exact
for conformal projections (polar stereographic, UTM, Transverse Mercator,
Lambert Conformal Conic), where k is isotropic so a single factor applies to
both axes; for a non-conformal projection the module warns, because the
distortion is then anisotropic. The implementation is shared with the
central-difference module so that the two products cannot diverge in this
respect.

The correction is OFF by default, so that the default output matches the
convention used by mainstream GIS slope tools, which work in nominal grid units.
Whenever it is off and the detected distortion exceeds 0.1 %, the module emits a
warning stating the measured k and the resulting bias, so the choice is never
made silently. If the two slope products are to be compared, the setting must
be the same for both.

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
cell contributes an elevation difference of about 10,099 m to a numerator that
should hold a few metres, producing a near-vertical slope where the terrain may
be flat. The error is not local either, since it propagates to every cell that
uses the sentinel as a neighbour. Treating NoData as zero elevation is the same
error in a less obvious form: it fabricates a cliff at the edge of every data
gap.

The rule applied here is strict and conservative:

    A cell receives a valid slope value only if ALL NINE cells of its 3x3
    neighbourhood are valid data. Otherwise the output cell is set to NoData.

The eight surrounding cells are required because all eight appear in Horn's
equations, so an incomplete window has no defined Horn estimate. The focal cell
is required too, even though it does not enter the arithmetic, because writing
a slope value where the DEM records no ground would assert terrain that was
never observed, and because the central-difference product applies the same
rule, which keeps the two comparable.

This means a NoData region in the DEM is dilated by exactly one cell in ALL
EIGHT directions in the slope product, including diagonally — a slightly larger
dilation than the central-difference product, whose gaps grow only along the
four cardinal directions. That is the intended and documented behaviour: it
guarantees that every slope value written to the output is computed from nine
genuine elevation observations and from Horn's formula alone, with no
substitution, no gap filling, no one-sided fallback, and no implicit
extrapolation across a data boundary. It is preferable for scientific use to
produce a smaller, fully defensible valid area than a complete raster
containing silently fabricated values.

Validity is determined from, in combination: the GDAL band mask (which honours
the declared nodata value and any internal or alpha mask), an explicit
comparison against the declared nodata value, and a finiteness test that
excludes NaN and +/- infinity.

## 4.1 Numerical audit of the output

Slope in degrees derived from the arctangent of a finite gradient is confined
to [0, 90). Values outside that interval, and NaN or infinite values inside the
valid area, are therefore mathematically impossible for a correct computation
and would indicate a defect in the implementation or a corrupted input rather
than an unusual landscape. The module counts them and reports them as an error;
it does not clip them, which would conceal the defect while leaving the product
wrong. The diagnosis to pursue, in order, is the DEM's NoData declaration, the
z-factor, and the cell size — a slope pinned near 90 degrees over a wide area
is the signature of a sentinel value entering the arithmetic.

## 4.2 Relationship to the machine-learning stage

Horn's method is a deterministic numerical procedure, not machine learning: the
same DEM, cell size and options always produce exactly the same slope raster,
and no parameter of the method is fitted to data. The machine-learning stage is
separate and occurs later, when this slope raster is used as a terrain
predictor alongside satellite-derived variables, or when it serves as the
reference target that a model built from those variables is trained to
reproduce. Where the model operates at a coarser resolution than the DEM — for
example a 10 m Sentinel-2 model against a 1 m lidar-derived slope raster — the
reference slope must be aggregated to the analysis scale and reported as such,
since a 10 m model does not reconstruct the slope of every 1 m lidar cell.
