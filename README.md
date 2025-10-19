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
```bash
git clone https://github.com/<your-username>/salmonberry-fusion.git
cd salmonberry-fusion
pip install -r requirements.txt
```

---

## 🛰️ Downloading Landsat Data

The first script focuses on retrieving Landsat imagery filtered by cloud coverage and date range.

```bash
python src/data/download_landsat.py     --region "Quinhagak"     --cloud 10     --start "2024-06-01"     --end "2024-08-30"
```

This will save filtered imagery to the `/data/raw` folder for preprocessing.

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
