# Configuration Integration Summary

## Overview
Updated `sentinel_query.py` to read configuration from `sentinel_query.yaml`. All configurable parameters are now centralized in the YAML file, making the code more maintainable and flexible.

## New Imports
- Added `Dict` and `Any` to typing imports
- Added `re` module for environment variable resolution
- Added `yaml` module import with graceful error handling

## New Functions

### Configuration Loading Functions

#### `_resolve_env_vars(value: Any) -> Any`
- **Purpose**: Recursively resolves environment variable references in config values
- **Format**: Supports `${VAR_NAME}` syntax (e.g., `"${SENTINEL_USER}"`)
- **Behavior**: Replaces with `os.environ.get()` value or empty string if not found
- **Handles**: Strings, dictionaries, and lists recursively

#### `load_config(config_path: str) -> Dict[str, Any]`
- **Purpose**: Loads and parses the YAML configuration file
- **Returns**: Dictionary containing all configuration sections
- **Features**:
  - Validates file existence
  - Logs configuration loading
  - Resolves environment variable references
  - Proper error handling with FileNotFoundError

### Query Functions with Config Support

#### `query_sentinel_products_from_config(config, wkt_area=None)`
- **Purpose**: Query Sentinel-2 products using YAML config parameters
- **Config Parameters Used**:
  - `query.product_type`
  - `query.platform_name`
  - `query.cloud_max`
  - `query.date_range.start_year`
  - `query.date_range.end_year`
  - `query.date_range.months`
  - `spatial.wkt_area` (can be overridden by function parameter)
- **Returns**: List of product metadata dictionaries

### Download Functions with Config Support

#### `download_products_from_config(config, products)`
- **Purpose**: Download products using output directory from YAML config
- **Config Parameters Used**:
  - `credentials.user`
  - `credentials.password`
  - `credentials.api_url`
  - `outputs.download_dir`
- **Returns**: List of downloaded file paths

### Polygon Creation Functions with Config Support

#### `create_polygons_from_config(config)`
- **Purpose**: Create polygon feature class from Emlid CSV using config settings
- **Config Parameters Used**:
  - `inputs.emlid_csv`
  - `inputs.lat_field`
  - `inputs.lon_field`
  - `outputs.study_areas_fc`
  - `spatial.sr_input_epsg`
  - `spatial.sr_output_epsg`
- **Returns**: Path to created feature class

### Band Compositing Functions with Config Support

#### `stack_rgb_from_config(config, sentinel_safe_dir, output_filename=None)`
- **Purpose**: Create RGB composite using band list from config
- **Config Parameters Used**:
  - `processing.rgb_bands`
  - `outputs.raster_output_dir`
- **Returns**: Path to created composite raster

#### `stack_nir_from_config(config, sentinel_safe_dir, output_filename=None)`
- **Purpose**: Create NIR composite using band list from config
- **Config Parameters Used**:
  - `processing.nir_bands`
  - `outputs.raster_output_dir`
- **Returns**: Path to created composite raster

## Enhanced Dataclass

### `SentinelCredentials`
Added class method:

#### `from_config(config: Dict[str, Any]) -> SentinelCredentials`
- **Purpose**: Factory method to create credentials from YAML config
- **Config Parameters Used**:
  - `credentials.user`
  - `credentials.password`
  - `credentials.api_url`
- **Returns**: SentinelCredentials instance

## Updated Existing Functions

### `query_sentinel_products()`
- **Added Parameters**:
  - `months: Tuple[int, int] = (5, 8)` - Allow custom month range
- **Enhanced Logging**: Now logs the month range used in queries

## Updated Main Example

The `main_example()` function now:

1. **Loads Configuration**: Reads from `sentinel_query.yaml` in the same directory
2. **Configures Logging**: Sets log level from `logging.level` config
3. **Step 1 - Create Study Area**: 
   - Reads Emlid CSV if `inputs.emlid_csv` is specified
   - Creates polygon feature class at `outputs.study_areas_fc`
4. **Step 2 - Query Products**:
   - Uses all query parameters from config (dates, cloud cover, product type)
   - Can use WKT from config or extracted from study area feature class
5. **Step 3 - Download Products**:
   - Respects `processing.limit_products` setting
   - Downloads to `outputs.download_dir`
6. **Step 4 - Create Composites**:
   - Creates RGB composites using `processing.rgb_bands`
   - Creates NIR composites using `processing.nir_bands`
   - Saves to `outputs.raster_output_dir`

## YAML Configuration Structure

The script now supports the complete YAML structure:

```yaml
credentials:
  user: "${SENTINEL_USER}"
  password: "${SENTINEL_PW}"
  api_url: "https://dataspace.copernicus.eu/"

query:
  product_type: "S2MSI2A"
  platform_name: "Sentinel-2"
  cloud_max: 5.0
  date_range:
    start_year: 2019
    end_year: 2024
    months: [5, 8]

spatial:
  sr_input_epsg: 4326
  sr_output_epsg: 4326
  wkt_area: null
  geojson_path: null

inputs:
  emlid_csv: null
  lat_field: "lat"
  lon_field: "lon"

outputs:
  download_dir: "./sentinel_downloads"
  gdb_path: "./data.gdb"
  study_areas_fc: "./data.gdb/study_areas"
  raster_output_dir: "./raster_composites"

processing:
  rgb_bands: ["B04", "B03", "B02"]
  nir_bands: ["B08", "B04", "B03"]
  limit_products: 3

logging:
  level: "INFO"
  file_path: "./sentinel_query.log"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

## Benefits

1. **Centralized Configuration**: All settings in one YAML file
2. **Environment Variables**: Secure credential handling via `${VAR_NAME}` syntax
3. **Backward Compatible**: Original functions still work with explicit parameters
4. **Flexible**: Easy to add new config-based wrapper functions
5. **Maintainable**: Reduced code duplication through config-aware functions
6. **Configurable Logging**: Log level controlled via YAML
7. **Easy Customization**: Non-developers can modify behavior without touching code

## Usage Examples

### Basic Usage with Config
```python
from sentinel_query import load_config, query_sentinel_products_from_config

config = load_config("sentinel_query.yaml")
products = query_sentinel_products_from_config(config)
```

### Run Complete Workflow
```python
from sentinel_query import main_example

main_example()  # Reads sentinel_query.yaml and runs full workflow
```

### Mix Config and Parameters
```python
config = load_config("sentinel_query.yaml")
# Override WKT area from function parameter
products = query_sentinel_products_from_config(
    config, 
    wkt_area="POLYGON ((...))"
)
```

## Installation Notes

Additional dependency required:
```bash
pip install pyyaml
```

The script will raise an informative error if PyYAML is not installed.
