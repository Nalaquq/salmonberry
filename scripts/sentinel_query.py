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
from typing import Iterable, List, Optional, Tuple, Dict, Any
import re

# External libs
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt  # pip install sentinelsat
import arcpy  # ArcGIS Pro provided
from arcpy import env
from arcpy import management as mgmt

try:
    import yaml
except ImportError:
    raise ImportError("PyYAML is required. Install with: pip install pyyaml")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# Configuration Loading
# ----------------------------

def _resolve_env_vars(value: Any) -> Any:
    """
    Recursively resolve environment variable references in config values.
    Handles strings like "${VAR_NAME}" by replacing with os.environ.get("VAR_NAME").
    """
    if isinstance(value, str):
        # Match ${VAR_NAME} pattern and replace with env var
        def replace_env(match):
            var_name = match.group(1)
            return os.environ.get(var_name, "")
        return re.sub(r"\$\{([^}]+)\}", replace_env, value)
    elif isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]
    return value


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and parse YAML configuration file.
    Resolves environment variable references (${VAR_NAME}).
    
    Args:
        config_path: Path to sentinel_query.yaml file
        
    Returns:
        Dictionary containing configuration
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    logger.info(f"Loading configuration from: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Resolve environment variable references
    config = _resolve_env_vars(config)
    logger.debug(f"Configuration loaded successfully")
    return config

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

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SentinelCredentials":
        """Create SentinelCredentials from YAML config dictionary."""
        creds_config = config.get("credentials", {})
        return cls(
            user=creds_config.get("user", ""),
            password=creds_config.get("password", ""),
            api_url=creds_config.get("api_url", "https://dataspace.copernicus.eu/")
        )

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
    months: Tuple[int, int] = (5, 8),
) -> List[dict]:
    """Query Copernicus for Sentinel-2 products meeting the criteria.

    - wkt_area: area-of-interest as WKT polygon or multipolygon
    - start_year, end_year: integer years (e.g., 2018, 2024)
    - cloud_max: maximum cloud cover percentage (e.g., 5.0)
    - product_type: Sentinel-2 product type (S2MSI2A or S2MSI1C)
    - platformname: satellite platform name
    - months: tuple of (start_month, end_month) for seasonal constraint
    - returns a list of product metadata dicts from sentinelsat.
    """
    logger.info("Connecting to Copernicus with sentinelsat at %s", creds.api_url)
    api = SentinelAPI(creds.user, creds.password, creds.api_url)

    results = []
    date_ranges = _month_limited_date_ranges(start_year, end_year, months=months)
    logger.info("Querying date ranges (months %d-%d) for years %d-%d", 
                months[0], months[1], start_year, end_year)

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


def query_sentinel_products_from_config(
    config: Dict[str, Any],
    wkt_area: Optional[str] = None,
) -> List[dict]:
    """
    Query Sentinel-2 products using parameters from YAML config.
    
    Args:
        config: Configuration dictionary from load_config()
        wkt_area: Override WKT area from config (optional)
        
    Returns:
        List of product metadata dictionaries
    """
    creds = SentinelCredentials.from_config(config)
    
    # Get spatial parameters
    spatial_cfg = config.get("spatial", {})
    if wkt_area is None:
        wkt_area = spatial_cfg.get("wkt_area")
    
    if not wkt_area:
        raise ValueError("No WKT area specified in config or parameters")
    
    # Get query parameters
    query_cfg = config.get("query", {})
    date_range = query_cfg.get("date_range", {})
    
    return query_sentinel_products(
        creds=creds,
        wkt_area=wkt_area,
        start_year=date_range.get("start_year", 2019),
        end_year=date_range.get("end_year", 2024),
        cloud_max=query_cfg.get("cloud_max", DEFAULT_CLOUD_MAX),
        product_type=query_cfg.get("product_type", SENTINEL2_PRODUCTTYPE),
        platformname=query_cfg.get("platform_name", "Sentinel-2"),
        months=tuple(date_range.get("months", [5, 8])),
    )


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


def download_products_from_config(
    config: Dict[str, Any],
    products: Iterable[dict],
) -> List[str]:
    """
    Download products using output directory from YAML config.
    
    Args:
        config: Configuration dictionary from load_config()
        products: List of product metadata dictionaries
        
    Returns:
        List of downloaded file paths
    """
    creds = SentinelCredentials.from_config(config)
    output_cfg = config.get("outputs", {})
    download_dir = output_cfg.get("download_dir", "./sentinel_downloads")
    
    return download_products(creds, products, download_dir)


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


def create_polygons_from_config(config: Dict[str, Any]) -> str:
    """
    Create polygon feature class from Emlid CSV using config settings.
    
    Args:
        config: Configuration dictionary from load_config()
        
    Returns:
        Path to created feature class
    """
    inputs_cfg = config.get("inputs", {})
    outputs_cfg = config.get("outputs", {})
    spatial_cfg = config.get("spatial", {})
    
    emlid_csv = inputs_cfg.get("emlid_csv")
    if not emlid_csv:
        raise ValueError("emlid_csv path not specified in config")
    
    return create_polygons_from_emlid(
        emlid_csv_path=emlid_csv,
        out_feature_class=outputs_cfg.get("study_areas_fc", "./data.gdb/study_areas"),
        lat_field=inputs_cfg.get("lat_field", "lat"),
        lon_field=inputs_cfg.get("lon_field", "lon"),
        sr_in_epsg=spatial_cfg.get("sr_input_epsg", 4326),
        sr_out_epsg=spatial_cfg.get("sr_output_epsg", 4326),
    )


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


def stack_rgb_from_config(
    config: Dict[str, Any],
    sentinel_safe_dir: str,
    output_filename: Optional[str] = None,
) -> str:
    """
    Create RGB composite using band list from config.
    
    Args:
        config: Configuration dictionary from load_config()
        sentinel_safe_dir: Path to unzipped Sentinel-2 SAFE directory
        output_filename: Override output filename (optional)
        
    Returns:
        Path to created composite raster
    """
    processing_cfg = config.get("processing", {})
    outputs_cfg = config.get("outputs", {})
    
    rgb_bands = processing_cfg.get("rgb_bands", ["B04", "B03", "B02"])
    raster_output_dir = outputs_cfg.get("raster_output_dir", "./raster_composites")
    
    if output_filename is None:
        output_filename = "composite_rgb.tif"
    
    out_raster = os.path.join(raster_output_dir, output_filename)
    _ensure_dir(raster_output_dir)
    
    return select_and_stack_bands(sentinel_safe_dir, rgb_bands, out_raster)


def stack_nir_from_config(
    config: Dict[str, Any],
    sentinel_safe_dir: str,
    output_filename: Optional[str] = None,
) -> str:
    """
    Create NIR composite using band list from config.
    
    Args:
        config: Configuration dictionary from load_config()
        sentinel_safe_dir: Path to unzipped Sentinel-2 SAFE directory
        output_filename: Override output filename (optional)
        
    Returns:
        Path to created composite raster
    """
    processing_cfg = config.get("processing", {})
    outputs_cfg = config.get("outputs", {})
    
    nir_bands = processing_cfg.get("nir_bands", ["B08", "B04", "B03"])
    raster_output_dir = outputs_cfg.get("raster_output_dir", "./raster_composites")
    
    if output_filename is None:
        output_filename = "composite_nir.tif"
    
    out_raster = os.path.join(raster_output_dir, output_filename)
    _ensure_dir(raster_output_dir)
    
    return select_and_stack_bands(sentinel_safe_dir, nir_bands, out_raster)


# ----------------------------
# Example usage inside ArcGIS Pro
# ----------------------------

def main_example():
    """
    Example script that:
    - Reads configuration from YAML file
    - Creates polygon study area from Emlid CSV
    - Queries Sentinel-2 L2A products using config parameters
    - Downloads products
    - Creates RGB composite
    """
    # Load configuration from YAML file
    config_path = os.path.join(os.path.dirname(__file__), "sentinel_query.yaml")
    config = load_config(config_path)
    
    # Configure logging from config
    logging_cfg = config.get("logging", {})
    log_level = logging_cfg.get("level", "INFO")
    logging.getLogger().setLevel(getattr(logging, log_level))
    
    inputs_cfg = config.get("inputs", {})
    outputs_cfg = config.get("outputs", {})
    processing_cfg = config.get("processing", {})
    
    # Step 1: Create polygons from Emlid CSV (if provided)
    if inputs_cfg.get("emlid_csv"):
        logger.info("Step 1: Creating study area polygons from Emlid CSV")
        create_polygons_from_config(config)
        study_areas_fc = outputs_cfg.get("study_areas_fc")
    else:
        logger.warning("No Emlid CSV specified in config. Using WKT area from config.")
        study_areas_fc = None
    
    # Step 2: Query Sentinel-2 products
    logger.info("Step 2: Querying Sentinel-2 products")
    spatial_cfg = config.get("spatial", {})
    wkt_area = spatial_cfg.get("wkt_area")
    
    if study_areas_fc and arcpy.Exists(study_areas_fc):
        # Use geometry from feature class
        geom = arcpy.da.SearchCursor(study_areas_fc, ["SHAPE@WKT"]).next()[0]
        wkt_area = geom
        logger.info("Using WKT area from study_areas feature class")
    
    if not wkt_area:
        raise ValueError("No WKT area available from config or study area feature class")
    
    products = query_sentinel_products_from_config(config, wkt_area=wkt_area)
    
    if not products:
        logger.warning("No products found matching criteria")
        return
    
    # Step 3: Download products
    logger.info("Step 3: Downloading products")
    limit = processing_cfg.get("limit_products")
    products_to_download = products[:limit] if limit else products
    
    downloaded = download_products_from_config(config, products_to_download)
    
    # Step 4: Create composites for downloaded products
    if downloaded:
        logger.info("Step 4: Creating RGB and NIR composites")
        for i, safe_dir in enumerate(downloaded):
            try:
                # RGB composite
                rgb_output = f"product_{i:02d}_rgb.tif"
                stack_rgb_from_config(config, safe_dir, rgb_output)
                logger.info("Created RGB composite: %s", rgb_output)
                
                # NIR composite
                nir_output = f"product_{i:02d}_nir.tif"
                stack_nir_from_config(config, safe_dir, nir_output)
                logger.info("Created NIR composite: %s", nir_output)
            except Exception as e:
                logger.exception("Failed to create composites for %s: %s", safe_dir, e)
    
    logger.info("Example workflow completed successfully")


if __name__ == "__main__":
    # Only run the example when executed directly (not when imported).
    main_example()
