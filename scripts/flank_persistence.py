r"""
flank_persistence.py
====================

Flank persistence F_d: how well the local gradient magnitude persists a fixed
horizontal distance uphill and downhill of each cell.

For every valid focal cell p with gradient (z_x, z_y) and magnitude
g0 = sqrt(z_x^2 + z_y^2) > minimum_gradient::

    u      = (z_x, z_y) / g0                 unit gradient direction (map axes)
    p_+    = p + d u                         straight-line uphill sample
    p_-    = p - d u                         straight-line downhill sample
    g_+/-  = bilinear sample of |grad Z| at p_+/-
    F_d    = min(g_+ / g0, g_- / g0)

The gradient components and magnitude are NOT recomputed here: they come from
``slope_central_differences.slope_central_differences_finite_difference``, so
F_d uses exactly the same central-difference stencil, NoData rule and sign
conventions as the slope product. The same module's ``validate_input`` and
``read_elevation_grid`` supply input validation and reading.

Only one raster is written: ``<output_directory>/flank_persistence.tif``.

Usage::

    python scripts/flank_persistence.py --config config/flank_persistence_config.yaml

See docs/flank_persistence.md for the method, NoData policy and limitations.
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

# The central-difference module lives beside this script; scripts/ is not a
# package, so make it importable regardless of the working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import slope_central_differences as cd  # noqa: E402

import rasterio  # noqa: E402  (availability already enforced by cd)
import yaml  # noqa: E402

LOGGER = logging.getLogger("flank_persistence")

#: Fixed output file name; the directory comes from the configuration.
OUTPUT_FILENAME = "flank_persistence.tif"

#: F_d >= 0 by construction, so any negative sentinel is unambiguous.
OUTPUT_NODATA: float = cd.DEFAULT_OUTPUT_NODATA
OUTPUT_DTYPE: str = "float32"

#: Fractional pixel offsets this close to an integer are snapped to it, so a
#: sample landing on a cell centre does not demand a zero-weight neighbour.
INTEGER_SNAP_TOLERANCE: float = 1e-9

CONFIG_KEYS = frozenset({
    "input_elevation_path",
    "output_directory",
    "jump_distance_m",
    "minimum_gradient",
})


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FlankConfig:
    """Validated, path-resolved contents of the YAML configuration."""

    input_elevation_path: str
    output_directory: str
    jump_distance_m: float
    minimum_gradient: float

    @property
    def output_path(self) -> str:
        return os.path.join(self.output_directory, OUTPUT_FILENAME)


def _as_number(value: object, key: str, config_path: str) -> float:
    """
    Return ``value`` as a finite float, rejecting booleans.

    Numeric strings are accepted because PyYAML (YAML 1.1) loads exponent
    forms without a dot or exponent sign, such as ``1e-8``, as strings.
    """
    if isinstance(value, bool):
        raise ValueError(
            f"'{key}' in {config_path} must be a number, got {value!r}."
        )
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"'{key}' in {config_path} must be a number, got {value!r}."
        ) from None
    if not math.isfinite(number):
        raise ValueError(f"'{key}' in {config_path} must be finite, got {value}.")
    return number


def load_config(config_path: str) -> FlankConfig:
    """
    Read and validate the flank-persistence YAML configuration.

    Relative paths resolve against the configuration file's directory, as in
    ``config/slope_config.yaml``. Unknown keys are an error, not a no-op.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")

    with open(config_path, "r", encoding="utf-8") as handle:
        try:
            document = yaml.safe_load(handle)
        except yaml.YAMLError as exc:
            raise ValueError(
                f"Configuration file is not valid YAML: {config_path}\n  {exc}"
            ) from exc

    if not isinstance(document, dict):
        raise ValueError(
            f"Configuration file must contain a mapping: {config_path}"
        )
    unknown = set(document) - CONFIG_KEYS
    if unknown:
        raise ValueError(
            f"Unrecognised key(s) in {config_path}: {', '.join(sorted(unknown))}\n"
            f"  Permitted keys: {', '.join(sorted(CONFIG_KEYS))}"
        )
    missing = CONFIG_KEYS - set(document)
    if missing:
        raise ValueError(
            f"Missing key(s) in {config_path}: {', '.join(sorted(missing))}"
        )

    config_dir = os.path.dirname(os.path.abspath(config_path))

    def _resolve(key: str) -> str:
        value = document[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"'{key}' in {config_path} must be a non-empty path.")
        expanded = os.path.expanduser(value)
        if not os.path.isabs(expanded):
            expanded = os.path.join(config_dir, expanded)
        return os.path.normpath(expanded)

    jump = _as_number(document["jump_distance_m"], "jump_distance_m", config_path)
    if jump <= 0.0:
        raise ValueError(
            f"'jump_distance_m' in {config_path} must be > 0, got {jump}."
        )
    min_grad = _as_number(document["minimum_gradient"], "minimum_gradient",
                          config_path)
    if min_grad < 0.0:
        raise ValueError(
            f"'minimum_gradient' in {config_path} must be >= 0, got {min_grad}."
        )

    return FlankConfig(
        input_elevation_path=_resolve("input_elevation_path"),
        output_directory=_resolve("output_directory"),
        jump_distance_m=jump,
        minimum_gradient=min_grad,
    )


def _validate_output_target(config: FlankConfig, overwrite: bool) -> None:
    """Refuse to write into a missing directory, over the input, or over an
    existing product unless ``overwrite`` is set."""
    if not os.path.isdir(config.output_directory):
        raise ValueError(
            f"Output directory does not exist: {config.output_directory}"
        )
    output = os.path.normcase(os.path.abspath(config.output_path))
    if output == os.path.normcase(os.path.abspath(config.input_elevation_path)):
        raise ValueError("The output path would overwrite the input DEM.")
    if os.path.exists(config.output_path) and not overwrite:
        raise ValueError(
            f"Output file already exists: {config.output_path}\n"
            "Pass --overwrite to replace it."
        )


# --------------------------------------------------------------------------
# Spatial helpers
# --------------------------------------------------------------------------


def jump_distance_in_map_units(crs: object, jump_distance_m: float) -> float:
    """
    Convert the configured distance in metres into the CRS's linear unit.

    Raises ValueError unless the CRS is projected and its linear unit's size
    in metres is known; degrees cannot be converted to a fixed distance.
    """
    if crs is None or crs.is_geographic or not crs.is_projected:
        raise ValueError(
            f"CRS {crs} is not a projected CRS with linear units; the jump "
            "distance cannot be expressed in its coordinates. Reproject the DEM."
        )
    unit_name, unit_metres = cd._describe_linear_unit(crs)
    if unit_metres is None or not math.isfinite(unit_metres) or unit_metres <= 0:
        raise ValueError(
            f"The size of the CRS linear unit '{unit_name}' in metres is "
            "unknown, so jump_distance_m cannot be converted to map units."
        )
    return jump_distance_m / unit_metres


def map_offsets_to_pixel_offsets(
    transform: object, dx_map: np.ndarray, dy_map: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert map-coordinate displacements into (row, column) displacements.

    A displacement is a vector, so only the linear part of the affine
    transform applies::

        [dx_map]   [a  b] [dcol]          [dcol]   [a  b]^-1 [dx_map]
        [dy_map] = [d  e] [drow]   ==>    [drow] = [d  e]    [dy_map]

    The general inverse is used, so this is correct for any non-singular
    transform; rotated/sheared grids are nevertheless rejected upstream by
    ``validate_input`` because the derivative stencil assumes axis alignment.
    """
    a, b, d, e = transform.a, transform.b, transform.d, transform.e
    det = a * e - b * d
    if det == 0.0:
        raise ValueError("Raster geotransform is singular.")
    dcol = (e * dx_map - b * dy_map) / det
    drow = (-d * dx_map + a * dy_map) / det
    return drow, dcol


def bilinear_sample(
    surface: np.ndarray, rows: np.ndarray, cols: np.ndarray
) -> np.ndarray:
    """
    Bilinearly sample ``surface`` at fractional (row, col) cell-centre indices.

    NoData policy: the result is NaN if the position lies outside the grid of
    cell centres, or if any neighbour carrying a non-zero interpolation weight
    is non-finite. Positions within ``INTEGER_SNAP_TOLERANCE`` of a cell row or
    column are snapped to it, so a sample on a cell centre needs only that cell
    and a sample on a row/column line needs only the two cells on that line.
    """
    n_rows, n_cols = surface.shape
    rows = np.asarray(rows, dtype=np.float64)
    cols = np.asarray(cols, dtype=np.float64)

    rows = np.where(np.abs(rows - np.round(rows)) < INTEGER_SNAP_TOLERANCE,
                    np.round(rows), rows)
    cols = np.where(np.abs(cols - np.round(cols)) < INTEGER_SNAP_TOLERANCE,
                    np.round(cols), cols)

    out = np.full(rows.shape, np.nan, dtype=np.float64)
    inside = ((rows >= 0) & (rows <= n_rows - 1)
              & (cols >= 0) & (cols <= n_cols - 1))
    if not np.any(inside):
        return out

    r = rows[inside]
    c = cols[inside]
    r0 = np.floor(r).astype(np.intp)
    c0 = np.floor(c).astype(np.intp)
    wr = r - r0
    wc = c - c0
    # A zero-weight neighbour is not required, which also keeps r0 + 1 inside
    # the array when r sits exactly on the last row (likewise for columns).
    r1 = np.where(wr > 0, r0 + 1, r0)
    c1 = np.where(wc > 0, c0 + 1, c0)

    v00 = surface[r0, c0]
    v01 = surface[r0, c1]
    v10 = surface[r1, c0]
    v11 = surface[r1, c1]

    neighbours_valid = (np.isfinite(v00) & np.isfinite(v01)
                        & np.isfinite(v10) & np.isfinite(v11))
    with np.errstate(invalid="ignore"):
        value = ((v00 * (1.0 - wc) + v01 * wc) * (1.0 - wr)
                 + (v10 * (1.0 - wc) + v11 * wc) * wr)
    out[inside] = np.where(neighbours_valid, value, np.nan)
    return out


# --------------------------------------------------------------------------
# Core computation
# --------------------------------------------------------------------------


def compute_flank_persistence(
    dz_dx: np.ndarray,
    dz_dy: np.ndarray,
    gradient_magnitude: np.ndarray,
    transform: object,
    jump_distance_map: float,
    minimum_gradient: float,
) -> np.ndarray:
    """
    Compute F_d = min(g_+ / g0, g_- / g0) for every eligible focal cell.

    ``dz_dx`` and ``dz_dy`` are the derivatives with respect to map x and map
    y (as returned by the central-difference module), so the unit direction
    and the offset ``d * u`` are in map coordinates and are converted to pixel
    offsets with the affine transform. Cells that are not eligible, or whose
    samples are invalid under :func:`bilinear_sample`'s NoData policy, are NaN.
    """
    g0 = gradient_magnitude
    focal = (np.isfinite(g0) & np.isfinite(dz_dx) & np.isfinite(dz_dy)
             & (g0 > minimum_gradient))
    result = np.full(g0.shape, np.nan, dtype=np.float64)
    if not np.any(focal):
        return result

    rows, cols = np.nonzero(focal)
    g0_f = g0[focal]
    ux = dz_dx[focal] / g0_f
    uy = dz_dy[focal] / g0_f

    drow, dcol = map_offsets_to_pixel_offsets(
        transform, jump_distance_map * ux, jump_distance_map * uy
    )

    # Straight-line offsets along the focal gradient direction (not a
    # terrain-following path).
    g_plus = bilinear_sample(g0, rows + drow, cols + dcol)
    g_minus = bilinear_sample(g0, rows - drow, cols - dcol)

    # NaN in either sample propagates through np.minimum to NaN.
    result[focal] = np.minimum(g_plus / g0_f, g_minus / g0_f)
    return result


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def save_flank_persistence(
    config: FlankConfig,
    persistence: np.ndarray,
    grid: "cd.RasterGrid",
    jump_distance_map: float,
) -> None:
    """Write F_d as float32, co-registered with the input DEM."""
    valid = np.isfinite(persistence)
    output_array = np.where(valid, persistence, OUTPUT_NODATA).astype(OUTPUT_DTYPE)

    profile = grid.profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype=OUTPUT_DTYPE,
        nodata=OUTPUT_NODATA,
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        width=grid.profile["width"],
        height=grid.profile["height"],
        transform=grid.transform,
        crs=grid.crs,
    )

    with rasterio.open(config.output_path, "w", **profile) as dst:
        dst.write(output_array, 1)
        dst.set_band_description(1, "Flank persistence F_d (dimensionless)")
        dst.update_tags(
            derived_product="Flank persistence",
            definition="F_d = min(g(p + d u) / g(p), g(p - d u) / g(p)), "
                       "u = grad Z / |grad Z|",
            gradient_method="Central finite differences "
                            "(slope_central_differences.py)",
            interpolation="bilinear on the gradient-magnitude surface",
            jump_distance_m=f"{config.jump_distance_m:.10g}",
            jump_distance_map_units=f"{jump_distance_map:.10g}",
            horizontal_linear_unit=grid.linear_unit,
            minimum_gradient=f"{config.minimum_gradient:.10g}",
            scale_factor_correction="not applied",
            nodata_treatment="NoData unless g0 is finite and > minimum_gradient "
                             "and both samples are inside the raster with all "
                             "non-zero-weight neighbours valid",
            source_dem=os.path.basename(config.input_elevation_path),
            valid_output_cells=str(int(np.count_nonzero(valid))),
            software="flank_persistence.py (rasterio + numpy)",
        )

    LOGGER.info("Flank persistence raster written: %s", config.output_path)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def run(config: FlankConfig, overwrite: bool = False) -> np.ndarray:
    """Validate, compute and write; returns the F_d array (NaN = NoData)."""
    LOGGER.info("Input DEM     : %s", config.input_elevation_path)
    LOGGER.info("Output        : %s", config.output_path)

    _validate_output_target(config, overwrite)
    cd.validate_input(config.input_elevation_path, allow_geographic_crs=False)

    # z_factor 1.0 and no scale-factor correction: F_d is a ratio of gradient
    # magnitudes, so a uniform vertical or horizontal rescaling cancels.
    grid = cd.read_elevation_grid(config.input_elevation_path)
    jump_map = jump_distance_in_map_units(grid.crs, config.jump_distance_m)
    LOGGER.info(
        "Jump distance : %.10g m = %.10g %s (%.4g x / %.4g y cells)",
        config.jump_distance_m, jump_map, grid.linear_unit,
        jump_map / grid.cell_size_x, jump_map / grid.cell_size_y,
    )

    gradients = cd.slope_central_differences_finite_difference(grid)
    persistence = compute_flank_persistence(
        gradients.dz_dx,
        gradients.dz_dy,
        gradients.gradient_magnitude,
        grid.transform,
        jump_map,
        config.minimum_gradient,
    )

    valid = persistence[np.isfinite(persistence)]
    if valid.size:
        LOGGER.info(
            "F_d over %d valid cells: min %.4g, median %.4g, max %.4g",
            valid.size, valid.min(), np.median(valid), valid.max(),
        )
    else:
        LOGGER.warning("No cell received a flank-persistence value.")

    save_flank_persistence(config, persistence, grid, jump_map)
    return persistence


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="flank_persistence.py",
        description="Compute flank persistence F_d from an elevation GeoTIFF.",
        epilog="Example:\n  python scripts/flank_persistence.py "
               "--config config/flank_persistence_config.yaml",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True, metavar="PATH",
                        help="YAML configuration file.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace an existing flank_persistence.tif.")
    parser.add_argument("--quiet", action="store_true",
                        help="Report warnings and errors only.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)-8s %(message)s",
        stream=sys.stdout,
    )

    try:
        run(load_config(args.config), overwrite=args.overwrite)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        return 1
    LOGGER.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
