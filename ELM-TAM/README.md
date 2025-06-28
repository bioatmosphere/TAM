# ELM-TAM: E3SM Land Model Integration

An implementation of the TAM (Transport and Absorptive roots with Mycorrhizal fungi) conceptual framework in the E3SM Land Model (ELM).

## Overview

This directory contains the full-scale implementation of TAM within the Community Earth System Model's land component. The ELM-TAM integration bridges the conceptual TAM framework with operational Earth system modeling capabilities.

## Directory Structure

### data/
Contains data processing, parameter estimation, and output analysis workflows:

- **parameter/**: Parameter estimation and sensitivity analysis
  - `src/tam_sensitivity.py` - Global sensitivity analysis using SALib
  - `src/tam_sites.py` - Site-specific parameter derivation
  - `data/` - Parameter files and site data
  - `figures/` - Sensitivity analysis visualizations

- **benchmark/** (Historical): Data synthesis pipeline for model benchmarking
  - Previously contained automated data processing from multiple sources
  - Main processing scripts have been removed after completion

## Key Features

- **Multi-scale Integration**: Site-level to global-scale TAM parameter estimation
- **Sensitivity Analysis**: Comprehensive parameter uncertainty quantification
- **FluxNet Integration**: Site-specific calibration using tower measurement data
- **Global Application**: Scalable parameter sets for Earth system modeling

## Scientific Context

ELM-TAM represents the operational implementation of the TAM conceptual framework published in Global Change Biology (2023). This implementation enables:

- Fine-root system complexity in Earth system models
- Mycorrhizal-explicit carbon and nutrient cycling
- Vertically-resolved root functional groups
- Transport vs. absorptive root differentiation

## Usage

### Parameter Estimation
```bash
# Navigate to parameter source directory
cd data/parameter/src/

# Run sensitivity analysis
uv run python tam_sensitivity.py

# Generate site-specific parameters
uv run python tam_sites.py
```

### Dependencies
See project root `pyproject.toml` for complete dependency list including:
- Scientific computing: numpy, pandas, xarray, netcdf4
- Analysis: scikit-learn, salib
- Visualization: matplotlib, seaborn

## Related Components

- **TAM/**: Original proof-of-concept study
- **TAM-Earth/**: Global root evolution modeling
- **TAM-Hydro/**: Root-water interaction processes

## External Integration

- **E3SM Land Model**: Full Earth system model integration
- **OLMT**: Parameter estimation and sensitivity analysis framework
- **FluxNet**: Observational data for model calibration and validation