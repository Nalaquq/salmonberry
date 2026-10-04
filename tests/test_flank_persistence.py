"""Synthetic-surface checks for scripts/flank_persistence.py.

Run with:  python -m pytest tests/test_flank_persistence.py
"""

import os
import sys

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import flank_persistence as fp  # noqa: E402
import slope_central_differences as cd  # noqa: E402

UTM = CRS.from_epsg(32604)  # UTM 4N, metres


def _grid(n_rows, n_cols, dx=1.0, dy=1.0):
    """North-up transform and local map coordinates (X, Y) of cell centres,
    centred so that X = Y = 0 at the middle cell."""
    transform = Affine(dx, 0.0, 500000.0, 0.0, -dy, 6600000.0)
    cols, rows = np.meshgrid(np.arange(n_cols), np.arange(n_rows))
    X = (cols - (n_cols - 1) / 2.0) * dx
    Y = -(rows - (n_rows - 1) / 2.0) * dy  # northing increases upward
    return transform, X, Y


def _write_dem(path, z, transform, crs=UTM, nodata=-9999.0):
    data = np.where(np.isfinite(z), z, nodata).astype("float64")
    with rasterio.open(
        path, "w", driver="GTiff", width=z.shape[1], height=z.shape[0],
        count=1, dtype="float64", crs=crs, transform=transform, nodata=nodata,
    ) as dst:
        dst.write(data, 1)
    return str(path)


def _persistence(tmp_path, z, transform, jump_m, min_grad=1e-8, crs=UTM):
    dem = _write_dem(tmp_path / "dem.tif", z, transform, crs)
    grid = cd.read_elevation_grid(dem)
    res = cd.slope_central_differences_finite_difference(grid)
    jump_map = fp.jump_distance_in_map_units(grid.crs, jump_m)
    return fp.compute_flank_persistence(
        res.dz_dx, res.dz_dy, res.gradient_magnitude,
        grid.transform, jump_map, min_grad,
    )


def test_plane_gives_one_and_edge_nodata(tmp_path):
    transform, X, Y = _grid(15, 17)
    z = 0.3 * X  # rises east, constant gradient 0.3
    F = _persistence(tmp_path, z, transform, jump_m=1.0)

    # Gradient valid on rows/cols 1..n-2; a 1-cell eastward jump from col 1
    # lands on col 0 (no gradient), so only cols 2..n-3 are valid.
    expected_valid = np.zeros_like(F, dtype=bool)
    expected_valid[1:-1, 2:-2] = True
    assert np.array_equal(np.isfinite(F), expected_valid)
    np.testing.assert_allclose(F[expected_valid], 1.0, rtol=1e-12)


def test_flat_surface_is_all_nodata(tmp_path):
    transform, X, _ = _grid(9, 9)
    F = _persistence(tmp_path, np.full(X.shape, 100.0), transform, jump_m=1.0)
    assert not np.any(np.isfinite(F))


def test_diagonal_quadratic_fractional_offsets(tmp_path):
    # z = c (X + Y)^2 -> central differences are exact: g = 2c*sqrt(2)*|s|,
    # s = X + Y, direction (1, 1)/sqrt(2). A jump d changes s by d*sqrt(2),
    # so F = (|s| - d*sqrt(2)) / |s|. d = 0.7 gives non-integer pixel
    # offsets, and bilinear interpolation of a linear field is exact.
    # A flipped y sign would sample along s = const and give F = 1.
    c, d = 0.05, 0.7
    transform, X, Y = _grid(41, 41)
    s = X + Y
    F = _persistence(tmp_path, c * s ** 2, transform, jump_m=d)

    with np.errstate(divide="ignore", invalid="ignore"):
        expected = (np.abs(s) - d * np.sqrt(2.0)) / np.abs(s)
    check = np.isfinite(F) & (np.abs(s) > d * np.sqrt(2.0) + 2.0)
    assert check.sum() > 500
    np.testing.assert_allclose(F[check], expected[check], rtol=1e-10)

    # s = 0 has zero gradient: below threshold -> NoData.
    assert not np.any(np.isfinite(F[s == 0]))


@pytest.mark.parametrize("axis", ["x", "y"])
def test_non_square_cells(tmp_path, axis):
    # dx = 2 m, dy = 0.5 m; a 3 m jump in x is 1.5 columns, a 0.75 m jump in
    # y is 1.5 rows. z = c * t^2 along one axis -> F = (|t| - d) / |t|.
    c = 0.1
    transform, X, Y = _grid(41, 41, dx=2.0, dy=0.5)
    t, d = (X, 3.0) if axis == "x" else (Y, 0.75)
    F = _persistence(tmp_path, c * t ** 2, transform, jump_m=d)

    with np.errstate(divide="ignore", invalid="ignore"):
        expected = (np.abs(t) - d) / np.abs(t)
    step = 2.0 if axis == "x" else 0.5
    check = np.isfinite(F) & (np.abs(t) > d + 2 * step)
    assert check.sum() > 100
    np.testing.assert_allclose(F[check], expected[check], rtol=1e-10)


def test_nodata_hole_propagates_to_samples(tmp_path):
    transform, X, _ = _grid(15, 15)
    z = 0.3 * X
    z[7, 7] = np.nan
    F = _persistence(tmp_path, z, transform, jump_m=1.0)

    # The hole removes the gradient at (7,7) and its four neighbours
    # (incomplete stencil). East/west 1-cell jumps from (7,5) .. (7,9) and
    # (6,7)/(8,7) hit invalid gradient cells or are themselves invalid.
    for cell in [(7, 5), (7, 6), (7, 7), (7, 8), (7, 9), (6, 7), (8, 7)]:
        assert not np.isfinite(F[cell]), cell
    # Rows 6 and 8 away from column 7 are unaffected.
    assert F[6, 5] == pytest.approx(1.0) and F[8, 9] == pytest.approx(1.0)


def test_bilinear_sample_policy():
    surface = np.array([[1.0, 2.0], [3.0, np.nan]])
    out = fp.bilinear_sample(surface, np.array([0.0, 0.0, 0.5, 0.0, -0.1]),
                             np.array([0.0, 0.5, 0.0, 1.0, 0.0]))
    np.testing.assert_allclose(out[:4], [1.0, 1.5, 2.0, 2.0])
    assert np.isnan(out[4])  # outside the raster
    # Any non-zero weight on the NaN neighbour -> NaN.
    assert np.isnan(fp.bilinear_sample(surface, np.array([0.5]),
                                       np.array([0.5]))[0])


def test_unit_conversion_and_geographic_rejection():
    ftus = CRS.from_epsg(2263)  # NY Long Island, US survey feet
    assert fp.jump_distance_in_map_units(ftus, 1.0) == pytest.approx(
        3.280833333, rel=1e-8)
    assert fp.jump_distance_in_map_units(UTM, 2.5) == 2.5
    with pytest.raises(ValueError):
        fp.jump_distance_in_map_units(CRS.from_epsg(4326), 1.0)


def test_pixel_offsets_use_full_affine():
    # 30-degree rotated grid with 2 m cells: a map vector along the rotated
    # column axis must map to a pure column step.
    th = np.radians(30.0)
    t = Affine(2 * np.cos(th), -2 * np.sin(th), 0.0,
               2 * np.sin(th), 2 * np.cos(th), 0.0)
    drow, dcol = fp.map_offsets_to_pixel_offsets(
        t, np.array([2 * np.cos(th)]), np.array([2 * np.sin(th)]))
    np.testing.assert_allclose([drow[0], dcol[0]], [0.0, 1.0], atol=1e-12)


def _write_config(path, dem, out_dir, **overrides):
    values = {
        "input_elevation_path": str(dem).replace("\\", "/"),
        "output_directory": str(out_dir).replace("\\", "/"),
        "jump_distance_m": 2.0,
        "minimum_gradient": 1.0e-8,
    }
    values.update(overrides)
    lines = [f"{k}: {v!r}" if isinstance(v, str) else f"{k}: {v}"
             for k, v in values.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def test_end_to_end_cli(tmp_path):
    # z = X^3 + 0.5 X: gradient magnitude has a minimum along X = 0, so both
    # samples there are steeper than the focal cell and F_d > 1.
    transform, X, _ = _grid(21, 21)
    dem = _write_dem(tmp_path / "dem.tif", 0.01 * X ** 3 + 0.5 * X, transform)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    cfg = _write_config(tmp_path / "cfg.yaml", dem, out_dir)

    assert fp.main(["--config", cfg, "--quiet"]) == 0
    assert os.listdir(out_dir) == ["flank_persistence.tif"]

    with rasterio.open(dem) as src, \
            rasterio.open(out_dir / "flank_persistence.tif") as dst:
        assert (dst.width, dst.height) == (src.width, src.height)
        assert dst.transform == src.transform and dst.crs == src.crs
        assert dst.dtypes[0] == "float32" and dst.nodata == -9999.0
        F = dst.read(1)
    assert F[10, 10] > 1.0
    assert F[0, 0] == -9999.0

    # Re-running without --overwrite refuses; with it, succeeds.
    assert fp.main(["--config", cfg, "--quiet"]) == 1
    assert fp.main(["--config", cfg, "--quiet", "--overwrite"]) == 0


def test_config_validation(tmp_path):
    transform, X, _ = _grid(5, 5)
    dem = _write_dem(tmp_path / "dem.tif", X, transform)
    bad = [
        {"jump_distance_m": 0},
        {"jump_distance_m": True},
        {"minimum_gradient": -1.0},
        {"extra_key": 1},
        {"jump_distance_m": "abc"},
    ]
    for i, override in enumerate(bad):
        cfg = _write_config(tmp_path / f"c{i}.yaml", dem, tmp_path, **override)
        with pytest.raises(ValueError):
            fp.load_config(cfg)


def test_geographic_dem_rejected(tmp_path):
    transform = Affine(0.001, 0.0, -150.0, 0.0, -0.001, 60.0)
    _, X, _ = _grid(9, 9)
    dem = _write_dem(tmp_path / "dem.tif", X, transform,
                     crs=CRS.from_epsg(4326))
    cfg = _write_config(tmp_path / "cfg.yaml", dem, tmp_path)
    assert fp.main(["--config", cfg, "--quiet"]) == 1
    assert not (tmp_path / "flank_persistence.tif").exists()
