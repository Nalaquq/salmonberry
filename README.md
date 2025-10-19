# Salmonberry Detection Using UAV–Satellite Fusion 🌿

A research project integrating **UAV imagery** and **satellite (Landsat)** data to classify and detect **salmonberry (Rubus spectabilis)** vegetation in sub-Arctic Alaska.  
This repository follows the recommended Python project structure from the [Python Guide](https://docs.python-guide.org/writing/structure/) and [PEP 8](https://peps.python.org/pep-0008/) conventions for readability, maintainability, and reproducibility.

---

## 🧭 Project Overview

The goal of this project is to develop a **fusion model** combining UAV-based orthomosaics and satellite remote-sensing data for improved vegetation classification accuracy in tundra environments.  
The workflow includes:
1. **Landsat data retrieval** (with cloud coverage filtering)  
2. **UAV imagery preprocessing** (orthorectification, resampling)  
3. **Feature extraction** (vegetation indices: NDVI, GCI, NDWI)  
4. **Model training and fusion** (e.g., Random Forest, CNN-based approaches)  
5. **Accuracy assessment and visualization**

---

## 📂 Repository Structure

salmonberry-fusion/
│
├── README.md # Project overview, setup, and usage instructions
├── LICENSE # License information
├── setup.py # Installation and packaging script
├── requirements.txt # Python dependencies
├── .gitignore # Files and directories to be ignored by Git
│
├── data/ # Input and output data (not tracked by Git)
│ ├── raw/ # Original UAV and satellite data (read-only)
│ ├── processed/ # Preprocessed or cleaned datasets
│ └── external/ # Any external datasets or shapefiles
│
├── notebooks/ # Jupyter notebooks for exploration and visualization
│ ├── 01_data_download.ipynb
│ ├── 02_preprocessing.ipynb
│ ├── 03_feature_extraction.ipynb
│ └── 04_modeling.ipynb
│
├── src/ # Source code for project modules
│ ├── init.py
│ ├── data/
│ │ ├── init.py
│ │ ├── download_landsat.py # Script to download Landsat data with cloud coverage filter
│ │ └── preprocess_uav.py
│ ├── features/
│ │ ├── init.py
│ │ └── vegetation_indices.py # Functions for NDVI, GCI, NDWI, etc.
│ ├── models/
│ │ ├── init.py
│ │ └── fusion_model.py # Model training and fusion algorithms
│ └── visualization/
│ ├── init.py
│ └── plot_maps.py # Visualization utilities
│
├── tests/ # Unit and integration tests
│ ├── init.py
│ ├── test_download.py
│ ├── test_indices.py
│ └── test_model.py
│
└── docs/ # Documentation (figures, API docs, references)
├── references.md
├── figures/
└── usage_examples.md


---

## ⚙️ Getting Started

### Prerequisites
- Python 3.10+  
- `gdal`, `rasterio`, `geopandas`, `scikit-learn`, `torch` / `tensorflow` (depending on model)  
- Access to Google Earth Engine or USGS EarthExplorer API (optional for Landsat retrieval)

### Installation
```bash
git clone https://github.com/<your-username>/salmonberry-fusion.git
cd salmonberry-fusion
pip install -r requirements.txt

**## 🧠 Code Style & Conventions**
- Follows PEP 8 for formatting.
- Docstrings follow PEP 257.
- Function names → snake_case; Classes → CamelCase.
- Include typing hints for all functions.
- Use Black or Ruff for linting and auto-formatting.

**## 🧪 Testing**
Tests use pytest:
- pytest tests/

**## 🛰️ Future Development**
- Add UAV–Landsat co-registration pipeline
- Implement CNN-based spectral–spatial fusion model
- Integrate GPS field validation
- Publish dataset and trained models

**## 📜 Acknowledgments**
This project is conducted in collaboration with the U.S. Department of Agriculture (USDA) and Hampden-Sydney College, focusing on remote-sensing applications for ecological monitoring in Alaska.
