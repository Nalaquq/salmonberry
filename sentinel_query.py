"""
sentinel_query.py
A PEP8-style module for querying and downloading Sentinel-2 products from the
Copernicus Open Access Hub / Data Space using sentinelsat, intended for import
and use inside ArcGIS Pro (arcpy available).

Main features:
- Query by extent (WKT), cloud cover (<= 5%), date range confined to May-August
  across multiple years, product type (L2A recommended for surface reflectance).
- Download matched products.
- Helpers to create study-area polygons from Emlid CSVs and ensure proper CRS.
- Helpers to composite selected bands (RGB, NIR) using arcpy.

Requirements:
- sentinelsat
- requests
- arcpy (ArcGIS Pro)
- shapely (optional but helpful for WKT; fallback to arcpy)
- tqdm (optional for progress)
"""

from __future__ import annotations

import csv
import datetime as dt
import logging
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

# External libs
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt  # pip install sentinelsat
import arcpy  # ArcGIS Pro provided
from arcpy import env
from arcpy import management as mgmt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# Constants & Band Mapping
# ----------------------------
SENTINEL2_PRODUCTTYPE = "S2MSI2A"  # L2A surface reflectance (recommended)
DEFAULT_CLOUD_MAX = 5.0  # percent

# Sentinel-2 band names (common): 10m: B02 (blue), B03 (green), B04 (red), B08 (NIR)
S2_BANDS = {
    "B01": "Aerosols (60m)",
    "B02": "Blue (10m)",
    "B03": "Green (10m)",
    "B04": "Red (10m)",
    "B05": "Red Edge 1 (20m)",
    "B06": "Red Edge 2 (20m)",
    "B07": "Red Edge 3 (20m)",
    "B08": "NIR (10m)",
    "B8A": "NIR narrow (20m)",
    "B09": "Water vapour (60m)",
    "B10": "Cirrus (60m)",
    "B11": "SWIR1 (20m)",
    "B12": "SWIR2 (20m)",
}

# ----------------------------
# Data classes
# ----------------------------

@dataclass
class SentinelCredentials:
    user: str
    password: str
    api_url: str = "https://dataspace.copernicus.eu/"  # Data Space / SciHub

# ----------------------------
# Utilities
# ----------------------------

def _ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _month_limited_date_ranges(start_year: int, end_year: int,
                               months: Tuple[int, int] = (5, 8)
                               ) -> List[Tuple[str, str]]:
    """
    Build list of (start, end) ISO strings for each year constrained to months.
    months default (5,8) => May 1 to Aug 31 inclusive each year.
    """
    ranges = []
    for y in range(start_year, end_year + 1):
        start = dt.date(y, months[0], 1).isoformat()
        # day=31 is safe for months 5-8
        end = dt.date(y, months[1], 31).isoformat()
        ranges.append((start, end))
    return ranges


# ----------------------------
# Core functions: query & download
# ----------------------------

def query_sentinel_products(
    creds: SentinelCredentials,
    wkt_area: str,
    start_year: int,
    end_year: int,
    cloud_max: float = DEFAULT_CLOUD_MAX,
    product_type: str = SENTINEL2_PRODUCTTYPE,
    platformname: str = "Sentinel-2",
) -> List[dict]:
    """Query Copernicus for Sentinel-2 products meeting the criteria.

    - wkt_area: area-of-interest as WKT polygon or multipolygon
    - start_year, end_year: integer years (e.g., 2018, 2024)
    - cloud_max: maximum cloud cover percentage (e.g., 5.0)
    - returns a list of product metadata dicts from sentinelsat.
    """
    logger.info("Connecting to Copernicus with sentinelsat at %s", creds.api_url)
    api = SentinelAPI(creds.user, creds.password, creds.api_url)

    results = []
    date_ranges = _month_limited_date_ranges(start_year, end_year, months=(5, 8))
    logger.info("Querying date ranges (May-August) for years %d-%d", start_year, end_year)

    for start_iso, end_iso in date_ranges:
        logger.debug("Querying %s to %s", start_iso, end_iso)
        # sentinelsat accepts date=(start, end) and cloudcoverpercentage=(0, cloud_max)
        products = api.query(
            wkt_area,
            date=(start_iso, end_iso),
            platformname=platformname,
            producttype=product_type,
            cloudcoverpercentage=(0, float(cloud_max)),
        )
        for uid, meta in products.items():
            meta["_uid"] = uid
            results.append(meta)

    # Sort results by sensing date (descending)
    results.sort(key=lambda m: m.get("beginposition") or m.get("ingestiondate"), reverse=True)
    logger.info("Found %d products matching criteria", len(results))
    return results


def download_products(
    creds: SentinelCredentials,
    products: Iterable[dict],
    out_dir: str,
    api: Optional[SentinelAPI] = None,
) -> List[str]:
    """
    Download given products (metadata dicts) to out_dir.
    Returns list of local file paths of downloaded archives.
    """
    _ensure_dir(out_dir)
    if api is None:
        api = SentinelAPI(creds.user, creds.password, creds.api_url)

    downloaded = []
    for meta in products:
        uid = meta.get("_uid")
        if not uid:
            logger.warning("Product metadata missing UID, skipping: %s", meta)
            continue
        # local_filename from api.get_product_odata or api.download call
        try:
            logger.info("Downloading product %s (%s)", meta.get("title"), uid)
            res = api.download(uid, directory_path=out_dir)
            # res is a dict with 'path' key when successful
            local_path = res.get("path") if isinstance(res, dict) else res
            downloaded.append(local_path)
        except Exception as e:
            logger.exception("Failed to download product %s: %s", uid, e)
    logger.info("Downloaded %d products", len(downloaded))
    return downloaded


# ----------------------------
# Emlid input -> ArcGIS polygons
# ----------------------------

def create_polygons_from_emlid(
    emlid_csv_path: str,
    out_feature_class: str,
    lat_field: str = "lat",
    lon_field: str = "lon",
    sr_in_epsg: int = 4326,
    sr_out_epsg: int = 4326,
) -> str:
    """
    Read an Emlid-formatted CSV with columns [lat, lon, optional name] and create a
    polygon feature class (or multipoint/convex hull per study area).

    - emlid_csv_path: path to CSV
    - out_feature_class: full path in geodatabase or shapefile path
    - sr_in_epsg: EPSG code of coordinate columns (assume WGS84 = 4326 by default)
    - sr_out_epsg: desired output spatial reference (reproject if needed)
    Returns path to created feature class.
    """
    logger.info("Creating polygons from Emlid CSV: %s", emlid_csv_path)
    # create an in-memory feature class of points, then aggregate/group as needed
    sr_in = arcpy.SpatialReference(sr_in_epsg)
    sr_out = arcpy.SpatialReference(sr_out_epsg)

    # temporary point FC
    temp_points = "in_memory/emlid_points"
    if arcpy.Exists(temp_points):
        arcpy.Delete_management(temp_points)

    arcpy.management.CreateFeatureclass(
        out_path=os.path.dirname(temp_points) or arcpy.env.scratchGDB,
        out_name=os.path.basename(temp_points),
        geometry_type="POINT",
        spatial_reference=sr_in
    )

    # Add fields for any attributes (optional)
    with open(emlid_csv_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        if lon_field not in reader.fieldnames or lat_field not in reader.fieldnames:
            raise ValueError(f"CSV must contain fields '{lat_field}' and '{lon_field}'")
        cur = arcpy.da.InsertCursor(temp_points, ["SHAPE@XY"] + reader.fieldnames)
        for row in reader:
            lat = float(row[lat_field])
            lon = float(row[lon_field])
            values = [ (lon, lat) ] + [ row[f] for f in reader.fieldnames ]
            cur.insertRow(values)
        del cur

    # Option A: create convex hull per group or single hull for all points. Here we create single hull.
    temp_polygons = "in_memory/emlid_polygons"
    if arcpy.Exists(temp_polygons):
        arcpy.Delete_management(temp_polygons)

    arcpy.management.Dissolve(temp_points, temp_polygons, multi_part="SINGLE_PART")
    # Project if CRS differs
    if sr_in_epsg != sr_out_epsg:
        mgmt.Project(temp_polygons, out_feature_class, sr_out)
    else:
        mgmt.CopyFeatures(temp_polygons, out_feature_class)

    # cleanup
    if arcpy.Exists(temp_points):
        arcpy.Delete_management(temp_points)
    if arcpy.Exists(temp_polygons):
        arcpy.Delete_management(temp_polygons)

    logger.info("Created polygon feature class: %s", out_feature_class)
    return out_feature_class


# ----------------------------
# Band stacking helper (ArcGIS)
# ----------------------------

def select_and_stack_bands(
    sentinel_safe_dir: str,
    band_list: List[str],
    out_raster: str,
) -> str:
    """
    Given a Sentinel SAFE directory (unzipped .SAFE) and a list of band names
    like ['B04', 'B03', 'B02'] create a composite raster suitable for ArcGIS.

    Uses arcpy to locate band files and CompositeBands.
    NOTE: band files are typically in GRANULE/.../IMG_DATA/R10m or R20m folders.
    """
    logger.info("Preparing band stack from: %s", sentinel_safe_dir)
    # find band file paths (simple heuristic)
    band_files = []
    for band in band_list:
        # Look for file containing band name (e.g., *_B04_10m.jp2 or *_B04.jp2)
        found = None
        for root, _, files in os.walk(sentinel_safe_dir):
            for f in files:
                if band.upper() in f and f.lower().endswith((".jp2", ".tif")):
                    found = os.path.join(root, f)
                    break
            if found:
                break
        if not found:
            raise FileNotFoundError(f"Could not find band {band} inside {sentinel_safe_dir}")
        band_files.append(found)

    logger.info("Compositing bands: %s -> %s", band_list, out_raster)
    mgmt.CompositeBands(band_files, out_raster)
    logger.info("Composite created: %s", out_raster)
    return out_raster


# ----------------------------
# Example usage inside ArcGIS Pro
# ----------------------------

def main_example():
    """
    Example script that:
    - Reads Emlid CSV, creates polygon study area
    - Queries Sentinel-2 L2A May-Aug for 2019-2024 with cloud <= 5%
    - Downloads first N products
    """
    creds = SentinelCredentials(user=os.environ.get("SENTINEL_USER", "user"),
                                password=os.environ.get("SENTINEL_PW", "pass"))
    # Example: read a local Emlid CSV and create polygon
    emlid_csv = r"C:\data\emlid_studyareas.csv"
    study_fc = r"C:\data\emlid_studyareas.gdb\study_areas"
    create_polygons_from_emlid(emlid_csv, study_fc, lat_field="lat", lon_field="lon",
                               sr_in_epsg=4326, sr_out_epsg=4326)

    # Use first polygon geometry as WKT for query
    geom = arcpy.da.SearchCursor(study_fc, ["SHAPE@WKT"]).next()[0]
    wkt = geom  # WKT string
    logger.info("Using WKT area for query (first feature).")

    products = query_sentinel_products(creds, wkt_area=wkt, start_year=2019, end_year=2024,
                                       cloud_max=5.0)
    # Download the first 3
    out_dir = r"C:\data\sentinel_downloads"
    downloaded = download_products(creds, products[:3], out_dir)

    # Example: stack bands for first downloaded SAFE (unzipped path)
    if downloaded:
        safe_dir = downloaded[0]
        # If file is .zip, user should unzip outside or code can be extended to unzip
        out_rgb = r"C:\data\stacks\product_rgb.tif"
        select_and_stack_bands(safe_dir, ["B04", "B03", "B02"], out_rgb)
        logger.info("Example finished. Composite at: %s", out_rgb)


if __name__ == "__main__":
    # Only run the example when executed directly (not when imported).
    main_example()
