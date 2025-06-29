# QuadratiK-Examples

This repository contains example code and resources for the QuadratiK package, as referenced in the associated ACM manuscript. 

## Contents

- `R-Examples.R`: R code examples for QuadratiK, including:
  - k-sample and two-sample kernel-based tests (wine, breast cancer, exoplanet, HIGGS datasets)
  - Bandwidth (h) selection algorithms
  - Uniformity testing on the sphere (earthquake and satellite data)
  - Poisson kernel-based clustering and validation (wireless data)
  - Data visualization for spatial datasets
- `Python-Examples.py`: Python equivalents of the R examples, including:
  - All statistical tests and clustering as above
  - Data preprocessing code for OneWeb Satelite data example
  - Usage of QuadratiK's Python API for kernel tests and clustering
- `R-Comparisons.R`: R code for comparing MMD and Energy statistics on the Breast Cancer and Exoplanet datasets, and for performing uniformity tests on Earthquake and Satellite data.
- `Python-Comparisons.py`: Python code for comparing MMD and Energy statistics on the Wine dataset.
- `Datasets/`: Example datasets used in the code (CSV, TXT)

## Required Packages

### R
- QuadratiK
- energy
- kernlab
- sphunif

### Python
- QuadratiK
- hyppo

## Usage

1. Clone the repository:
   ```sh
   git clone https://github.com/rmj3197/QuadratiK-Examples
   cd QuadratiK-Examples
   ```

2. Install the QuadratiK package in your R or Python environment as required.

3. Run the examples:
   - For R: Open and run `R-Examples.R` or `R-Comparisons.R` in RStudio or your preferred R environment.
   - For Python: Run `Python-Examples.py` or `Python-Comparisons.py` in your Python environment.

4. Ensure all required datasets are present in the `Datasets/` folder. For large datasets (e.g., HIGGS), follow the instructions in the code comments to download them manually.

## Notes

- The examples are intended for demonstration and reproducibility. For details on each example, refer to the manuscript and code comments.
- Some code (e.g., HIGGS dataset) is commented out due to large data requirements or HPC-specific instructions.