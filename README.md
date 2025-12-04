# 🛰️ UAV–Satellite Fusion for Salmonberry Detection in Alaska

This repository hosts a research project focused on **fusing UAV and Landsat satellite data** to improve **vegetation classification**, with a particular focus on detecting *salmonberry (Rubus spectabilis)* in sub-Arctic Alaska.

The project integrates geospatial data processing, feature extraction (NDVI, GCI, NDWI), and model fusion techniques to enhance vegetation mapping accuracy. It adheres to the [Python Guide’s recommended structure](https://docs.python-guide.org/writing/structure/) and follows [PEP 8](https://peps.python.org/pep-0008/) and [PEP 257](https://peps.python.org/pep-0257/) coding conventions.

---

## 📖 Overview

Vegetation classification in sub-Arctic regions presents challenges due to cloud cover, sparse vegetation, and limited UAV coverage.  
This project aims to address these challenges by:
- **Combining UAV orthomosaics** with **Landsat multispectral imagery**  
- **Extracting vegetation indices** for improved class separability  
- **Building a fusion model** to detect salmonberry patches efficiently  

The research supports ecosystem monitoring and resource management in collaboration with the **USDA**.

---

## 🧭 Key Objectives
1. Develop an automated Landsat data retrieval script with cloud filtering.  
2. Fuse UAV and satellite data using spatial and spectral alignment.  
3. Train and evaluate machine learning and deep learning models for vegetation classification.  
4. Produce interpretable and reproducible visualizations for analysis.

---

## ⚙️ Project Structure

```
salmonberry-fusion/
│
├── README.md               # Project overview, setup, and usage
├── LICENSE                 # License details
├── requirements.txt        # Project dependencies
├── setup.py                # Package setup script
├── .gitignore              # Ignored files and directories
│
├── data/                   # Data storage (excluded from version control)
│   ├── raw/                # Original UAV and Landsat data
│   ├── processed/          # Cleaned and aligned datasets
│   └── external/           # Ancillary datasets (e.g., shapefiles)
│
├── scripts/                # Standalone scripts and utilities
│   ├── sentinel_query.py       # Query and download Sentinel-2 satellite imagery
│   ├── sentinel_query.yaml     # Configuration file for sentinel_query.py
│   └── CONFIG_INTEGRATION_SUMMARY.md  # Documentation for config integration
│
├── notebooks/              # Jupyter notebooks for analysis and visualization
│   ├── 01_data_download.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_extraction.ipynb
│   └── 04_modeling.ipynb
│
├── src/                    # Source code
│   ├── __init__.py
│   ├── data/
│   │   ├── download_landsat.py    # Script to fetch Landsat data by cloud coverage
│   │   └── preprocess_uav.py
│   ├── features/
│   │   └── vegetation_indices.py  # NDVI, GCI, NDWI, etc.
│   ├── models/
│   │   └── fusion_model.py        # Fusion and classification models
│   └── visualization/
│       └── plot_maps.py           # Plotting and visualization utilities
│
├── tests/                  # Unit and integration tests
│   ├── test_download.py
│   ├── test_indices.py
│   └── test_model.py
│
└── docs/                   # Documentation and references
    ├── references.md
    ├── usage_examples.md
    └── figures/
```

---

## 🚀 Getting Started

### Prerequisites
Make sure you have:
- Python **3.10+**
- Libraries: `rasterio`, `geopandas`, `gdal`, `numpy`, `scikit-learn`, `torch`, `matplotlib`
- Access to **USGS EarthExplorer API** or **Google Earth Engine** (for Landsat data)

### Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/salmonberry-fusion.git
cd salmonberry-fusion
```

#### 2. Create a Virtual Environment
A virtual environment isolates project dependencies from your system Python installation.

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\env\Scripts\Activate.ps1
```

**On Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the beginning of your terminal prompt when activated.

#### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Verify Installation
```bash
python -c "import sentinelsat; print('sentinelsat installed successfully')"
```

#### Deactivating the Virtual Environment
When finished working on the project, deactivate the virtual environment:
```bash
deactivate
```

---

## 🛰️ Downloading Sentinel-2 Satellite Data

This project includes a powerful script for querying and downloading Sentinel-2 imagery from the Copernicus Data Space Hub. All configuration is managed through a simple **YAML file**, making it easy to customize parameters without modifying code.

### Quick Start

1. **Configure Your Settings** - Edit `scripts/sentinel_query.yaml`:
   ```yaml
   credentials:
     user: "${SENTINEL_USER}"      # Set environment variable or use plain text
     password: "${SENTINEL_PW}"    # Set environment variable or use plain text
     api_url: "https://dataspace.copernicus.eu/"

   query:
     product_type: "S2MSI2A"       # L2A surface reflectance (recommended)
     cloud_max: 5.0                # Maximum cloud cover (%)
     date_range:
       start_year: 2019
       end_year: 2024
       months: [5, 8]              # May-August

   spatial:
     sr_input_epsg: 4326
     sr_output_epsg: 4326
     shapefile_path: null          # Path to .shp file (HIGHEST PRIORITY)
     geojson_path: null            # Path to GeoJSON file (2nd priority)
     wkt_area: null                # WKT polygon string (3rd priority)

   inputs:
     emlid_csv: null               # Path to Emlid survey CSV
     lat_field: "lat"
     lon_field: "lon"
     shapefile_feature_index: null # Feature index to use from shapefile (null = merge all)

   outputs:
     download_dir: "./sentinel_downloads"
     gdb_path: "./data.gdb"
     study_areas_fc: "./data.gdb/study_areas"
     raster_output_dir: "./raster_composites"

   processing:
     rgb_bands: ["B04", "B03", "B02"]    # Red, Green, Blue
     nir_bands: ["B08", "B04", "B03"]    # NIR, Red, Green
     limit_products: 3
   ```

2. **Run the Script** - Execute the complete workflow:
   ```bash
   cd scripts
   python sentinel_query.py
   ```

   The script will:
   - Load configuration from `sentinel_query.yaml`
   - Create study area polygons from Emlid CSV (if provided)
   - Query Sentinel-2 products matching your criteria
   - Download matched products
   - Create RGB and NIR composite rasters

### Configuration Details

- **Credentials**: Use environment variables (`${SENTINEL_USER}`, `${SENTINEL_PW}`) for secure credential management
- **Query Parameters**: Customize product type, cloud cover threshold, date range, and seasonal constraints
- **Spatial Settings**: Define coordinate reference systems and area-of-interest using:
  - **Shapefile** (`.shp`) - HIGHEST PRIORITY: Specify `shapefile_path` with optional `shapefile_feature_index`
  - **GeoJSON** - 2nd priority: Specify `geojson_path`
  - **WKT** - 3rd priority: Specify `wkt_area` as a polygon string
- **Inputs**: Provide Emlid CSV file paths for automated study area creation (if no shapefile provided)
- **Outputs**: Specify directories for downloads and processed rasters
- **Processing**: Select specific band combinations for composites (RGB, NIR, etc.)

### Using Shapefiles for Study Area Definition

Shapefiles are the **recommended** way to define your study area:

```yaml
spatial:
  shapefile_path: "./data/study_area.shp"  # Path to your shapefile
  sr_output_epsg: 4326                     # Output CRS (WGS84)

inputs:
  shapefile_feature_index: null            # null = use all features, 0 = first feature only
```

The script will:
- Read all features from the shapefile (or a specific feature by index)
- Automatically reproject to the specified output CRS
- Merge multiple features if needed
- Convert to WKT format for Sentinel-2 queries

### Programmatic Usage

You can also use the script as a Python module:

```python
from sentinel_query import load_config, query_sentinel_products_from_config, download_products_from_config

# Load configuration
config = load_config("sentinel_query.yaml")

# Query products
products = query_sentinel_products_from_config(config, wkt_area="POLYGON (...)")

# Download products
downloaded = download_products_from_config(config, products)

# Create composites
from sentinel_query import stack_rgb_from_config, stack_nir_from_config
for safe_dir in downloaded:
    stack_rgb_from_config(config, safe_dir, "output_rgb.tif")
    stack_nir_from_config(config, safe_dir, "output_nir.tif")
```

For detailed configuration integration documentation, see `scripts/CONFIG_INTEGRATION_SUMMARY.md`.

---


## 🧠 Code Style and Standards

This project adheres to:
- **PEP 8** – for consistent, readable Python code  
- **PEP 257** – for clear and concise docstrings  
- **Type Hints** – for function signatures and better IDE support  
- **Black / Ruff** – for code formatting and linting  

---

## 🧪 Testing

All modules are tested using **pytest**:

```bash
pytest tests/
```

---

## 📚 Documentation

- Each function and module includes **Google-style docstrings**
- Full documentation generated with **Sphinx**
- References and methodology stored under `/docs/`

---

## 🌱 Future Directions

- Integrate deep learning models (CNNs or transformers) for spectral–spatial fusion  
- Expand to other tundra vegetation classes  
- Incorporate GPS ground-truth validation  
- Publish an open dataset and trained model weights  

---

## 🤝 Acknowledgments

This work is conducted in collaboration with the **U.S. Department of Agriculture (USDA)** and **Hampden-Sydney College**, supporting research in environmental monitoring and vegetation mapping in Alaska.

---

## 👤 Authors

**Gyabaah Kyere**  
B.S. Candidate in Computer Science & Applied Mathematics  
Hampden-Sydney College | USDA Research Fellow  
📧 [kyeregyeabourg27@hsc.edu]  
🔗 [github.com/kyere7](https://github.com/kyere7)
