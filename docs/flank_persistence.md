# Flank persistence ($F_d$)

`scripts/flank_persistence.py` computes flank persistence, a custom terrain metric, from an
elevation GeoTIFF. It writes exactly one raster, `flank_persistence.tif`. It does not
modify `slope_central_differences.py`, its configuration or its outputs.

## Configure and run

Edit `config/flank_persistence_config.yaml`:

```yaml
input_elevation_path: "path/to/input_dem.tif"
output_directory: "path/to/output_directory"
jump_distance_m: 1.0        # horizontal sampling distance d, metres (> 0)
minimum_gradient: 1.0e-8    # focal cells need g0 > this (>= 0)
```

Then run:

```
python scripts/flank_persistence.py --config config/flank_persistence_config.yaml
```

The run writes `<output_directory>/flank_persistence.tif`.

- Relative paths resolve against the config file's directory, the same convention as
  `config/slope_config.yaml`.
- Unknown or missing keys are errors.
- The output directory must already exist.
- An existing `flank_persistence.tif` is not replaced unless you pass `--overwrite`.
- `--quiet` shows only warnings and errors.

Tests on synthetic surfaces: `python -m pytest tests/test_flank_persistence.py`.

## Definition

The gradient components $z_x$ and $z_y$ and the magnitude $g_0=\sqrt{z_x^2+z_y^2}$ come directly from
`slope_central_differences_finite_difference`, so they use the same second-order 4-neighbour
stencil, the same NoData rule and the same map-axis sign convention as the slope product.

For each focal cell $p$ where $g_0$ is finite and $g_0 > $ `minimum_gradient`:

$$u=(z_x,z_y)/g_0,\qquad p_\pm = p \pm d\,u$$

$$g_\pm = \text{bilinear sample of } g \text{ at } p_\pm,\qquad
\boxed{F_d=\min\!\left(\frac{g_+}{g_0},\frac{g_-}{g_0}\right)}$$

The samples are straight-line offsets along the focal gradient direction. They do not follow
the terrain. Only $F_d$ is written; $g_+$ and $g_-$ stay internal.

## Interpretation

$F_d$ is the weaker of the two directional persistence ratios.

| Value of $F_d$ | Meaning |
| --- | --- |
| $F_d \approx 1$ | The gradient magnitude a distance $d$ uphill and downhill is about the same as at the focal cell, for example on a uniform planar flank. |
| $F_d < 1$ | The gradient weakens on at least one side, for example near a crest, toe, bench or break of slope. |
| $F_d > 1$ | Both sides are steeper than the focal cell, for example in a local gradient minimum within a slope. The value is not clipped. |

**$F_d$ measures relative persistence of steepness.** It does not measure absolute steepness:
a gentle uniform slope and a steep uniform slope both give $F_d \approx 1$. Interpret it together with
slope or $g_0$.

**$F_d$ is not a measure of salmonberry abundance or distribution.** Any link to habitat is a
hypothesis that needs to be tested against field data.

## Parameters

- **`jump_distance_m` ($d$)**: the horizontal distance from the focal cell to each sample.
  - It is converted from metres to the CRS linear unit. For example, 1 m becomes 3.2808 ft in a
    US-survey-foot CRS.
  - Small $d$, around one cell or less, samples cells whose gradients share stencil cells with the
    focal gradient. Persistence is then mostly a measure of smoothness.
  - Larger $d$ tests persistence across a landform at that scale and loses more cells near edges
    and NoData.
  - Each run logs $d$ in cells.
- **`minimum_gradient`**: cells with $g_0$ at or below this value (flat ground, where the direction
  $u$ is undefined or dominated by noise) get NoData. The default value only guards against division
  by zero. Raise it, for example to `0.01`, to exclude near-flat ground where the direction is
  unreliable.

## CRS, interpolation and NoData

- **CRS.** The DEM must use a projected CRS whose linear unit has a known size in metres.
  - Geographic (degree) CRSs are rejected, as are CRSs with an unknown unit.
  - Rotated or sheared geotransforms are rejected. This check is inherited from
    `validate_input`, because the central-difference stencil assumes axis-aligned rows and columns.
  - The map-to-pixel offset uses the inverse of the affine transform's linear part, so non-square
    cells and south-up grids are handled.
- **Units.** The vertical and horizontal units are assumed to match (z-factor 1).
  - $F_d$ is a ratio of gradient magnitudes, so a uniform vertical or horizontal rescaling cancels out.
  - However, `minimum_gradient` is applied to the unscaled gradient.
- **Distance.** No projection scale-factor correction is applied. $d$ is a map-plane distance.
  - For UTM this is within about 0.1% of ground distance.
  - For ArcticDEM (EPSG:3413) near 60°N, 1 map metre is about 0.96 ground metres.
- **Interpolation.** Bilinear interpolation on the grid of cell centres.
- **NoData.** The output is NoData (-9999) when any of these is true:
  - The focal gradient is non-finite or not above the threshold. This includes the outer one-cell
    ring and any cell with an incomplete 5-cell stencil.
  - Either sample point falls outside the cell-centre grid.
  - Any of the up to four interpolation neighbours that has a non-zero weight has no valid
    gradient. A sample that lands exactly on a cell row, column or centre needs only the
    neighbours on that line or that point.
- **Output.** Float32, with the input's dimensions, CRS, transform, extent and alignment. Provenance
  tags record $d$, the threshold and the method.
- **Memory.** The whole raster is processed in memory, the same as the slope script. There is no
  windowing, so there are no window-boundary artefacts. Very large DEMs need enough RAM for
  several float64 copies.

## Limitations

- This is a custom metric with no published validation. Its ecological meaning is untested.
- It uses a single scale $d$ and only the focal direction. Curved flow lines or flanks that bend
  within distance $d$ are sampled off the actual slope line.
- It inherits central-difference sensitivity to DEM noise. Ratios amplify noise where $g_0$ is small.
- It is sensitive to surface type: a DSM such as ArcticDEM measures the canopy top, not the ground.
- More cells are lost near edges and NoData as $d$ grows.
