# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a multi-component ecosystem modeling system implementing the **TAM (Transport and Absorptive roots with Mycorrhizal fungi)** conceptual framework for terrestrial ecosystem modeling. The repository bridges ecological theory with practical Earth system modeling through comprehensive data synthesis and machine learning approaches.

## Architecture

The codebase consists of four main components:

1. **TAM/** - Original proof-of-concept study with simple_ELM integration
2. **ELM-TAM/** - Full E3SM Land Model integration with automated benchmarking pipeline
3. **TAM-Earth/** - Global-scale root evolution and distribution modeling  
4. **TAM-Hydro/** - Root-water interactions and hydrological processes

## Key Dependencies

- **Scientific computing:** numpy, pandas, xarray, netcdf4, matplotlib, seaborn
- **Geospatial:** gdal, geopandas, cartopy, contextily
- **Machine learning:** scikit-learn, salib (sensitivity analysis)
- **Data processing:** requests, tqdm, openpyxl, pdfplumber

## Primary Workflows

### ELM-TAM Benchmark Pipeline
The main development focus is in `ELM-TAM/data/benchmark/src/`, which implements an automated data synthesis pipeline:

- **Entry point:** `main.py` - Master pipeline script
- **Data sources:** ForC, ILAMB, FluxNet, TerraClimate, SoilGrids
- **Processing chain:** Raw data → standardization → integration → ML training → global application
- **Models:** Random Forest (`RF_model.py`) and Multi-Layer Perceptron (`MLP_model.py`)

### Parameter Estimation
Located in `ELM-TAM/data/parameter/src/`:
- `tam_sensitivity.py` - Sensitivity analysis using SALib
- `tam_sites.py` - Site-specific parameter derivation

### Data Integration Scripts
- `GherardiSala_data_2020.py` - Grassland productivity data
- `data_integration.py` - Multi-source data synthesis
- `soilgrids.py` - Global soil carbon data processing
- `ancillary_data.py` - Climate and environmental data

## Development Environment

Uses uv for dependency management with Python >=3.9. Virtual environment located in `.venv/`.

## Data Architecture

**Raw Data Sources:**
- FluxNet tower measurements
- ILAMB benchmarking datasets  
- ForC global forest carbon database
- TerraClimate gridded climate data
- SoilGrids global soil properties

**Data Flow:**
Raw data → Processing scripts → Standardized formats → Integration → Model training → Analysis/Visualization

## Scientific Context

This is a research-oriented codebase supporting the TAM conceptual framework published in Global Change Biology (2023). The work focuses on fine-root system complexity in terrestrial ecosystem models, with extensive literature review and empirical validation.

## External Integrations

- **simple_ELM:** Root complexity implementation (branch: rootcomplexity)
- **OLMT:** Parameter estimation and sensitivity analysis
- **E3SM Land Model:** Full Earth system model integration

## Working with the Code

1. Start with the main README and published papers to understand TAM concepts
2. Use `ELM-TAM/data/benchmark/src/main.py` as the primary workflow entry point
3. Leverage existing data processing scripts rather than recreating functionality
4. Follow scientific workflow - understand literature context before modifying code
5. Pay attention to data provenance - scripts depend on specific data sources and formats