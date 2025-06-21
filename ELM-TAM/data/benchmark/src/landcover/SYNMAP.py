"""
Download and process SYNMAP landcover data.

SYNMAP is a global land cover dataset that provides fractional coverage of plant functional types.
It's particularly useful for ecosystem modeling and carbon cycle studies.

Source: https://efmkoene.github.io/2021-10-11-SYNMAP_landcover_processing/
Data: https://www.bgc-jena.mpg.de/geodb/projects/Data.php

https://efmkoene.github.io/2021-10-11-SYNMAP_landcover_processing/

Reference: 
Jung, M., Henkel, K., Herold, M., & Churkina, G. (2006). 
Exploiting synergies of global land cover products for carbon cycle modeling. 
Remote Sensing of Environment, 101(4), 534-553.
https://doi.org/10.1016/j.rse.2006.01.020

Dependencies:
    pip install requests tqdm xarray rasterio matplotlib cartopy
"""

import os
import requests
from tqdm import tqdm
from pathlib import Path
import argparse
import zipfile
import tarfile
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

try:
    import xarray as xr
except ImportError:
    xr = None

try:
    import rasterio
except ImportError:
    rasterio = None


# SYNMAP data URLs and file configurations
# Note: SYNMAP data is available from multiple sources, some may require registration
SYNMAP_DATA = {
    "synmap_geotiff": {
        "url": "https://drive.google.com/file/d/1dj0uB5fqunbBoIeVbt-8BFzojqV2Bcbd/view?usp=sharing",
        "filename": "synmap_processed.tif",
        "description": "SYNMAP processed GeoTIFF from efmkoene (Google Drive - manual download required)",
        "note": "Visit URL manually to download from Google Drive. Cite original SYNMAP authors if used."
    },
    "synmap_original": {
        "url": "https://www.bgc-jena.mpg.de/bgc-systems/pmwiki2/pmwiki.php/Download/VPRMpreproc",
        "filename": "synmap_original.dat",
        "description": "SYNMAP original binary data from BGC Jena (manual download required)",
        "note": "Visit URL manually to download original binary file"
    },
    "databasin_synmap": {
        "url": "https://databasin.org/datasets/112a942ec4294e5284e63d5e6bf14b29/",
        "filename": "synmap_databasin.zip",
        "description": "SYNMAP from Data Basin (may require manual download)",
        "note": "Visit URL manually to download - registration may be required"
    },
    "earthenv_class1": {
        "url": "http://data.earthenv.org/consensus_landcover/with_DISCover/consensus_full_class_1.tif",
        "filename": "earthenv_consensus_class_1.tif",
        "description": "EarthEnv Consensus Land Cover - Class 1 (Evergreen/Deciduous Needleleaf Trees)"
    },
    "earthenv_class2": {
        "url": "http://data.earthenv.org/consensus_landcover/with_DISCover/consensus_full_class_2.tif", 
        "filename": "earthenv_consensus_class_2.tif",
        "description": "EarthEnv Consensus Land Cover - Class 2 (Evergreen Broadleaf Trees)"
    },
    "modis_landcover": {
        "url": "https://e4ftl01.cr.usgs.gov/MOTA/MCD12Q1.006/2020.01.01/MCD12Q1.A2020001.h00v08.006.2021365205006.hdf",
        "filename": "modis_landcover_sample.hdf",
        "description": "MODIS Land Cover sample tile (requires EARTHDATA login)"
    }
}

# SYNMAP biome type mapping (48 types, 0-47)
# Based on SYNMAP global potential vegetation dataset
BIOME_MAPPING = {
    0: "Tropical Evergreen Broadleaf Forest",
    1: "Tropical Semi-Evergreen Broadleaf Forest", 
    2: "Tropical Deciduous Broadleaf Forest",
    3: "Tropical Woodland",
    4: "Tropical Wooded Grassland",
    5: "Tropical Grassland",
    6: "Warm Temperate Evergreen Broadleaf & Mixed Forest",
    7: "Warm Temperate Deciduous Broadleaf Forest",
    8: "Warm Temperate Open Woodland",
    9: "Warm Temperate Wooded Grassland",
    10: "Warm Temperate Grassland",
    11: "Cool Temperate Mixed Forest",
    12: "Cool Temperate Conifer Forest",
    13: "Cool Temperate Deciduous Broadleaf Forest",
    14: "Cool Temperate Open Woodland",
    15: "Cool Temperate Wooded Grassland",
    16: "Cool Temperate Grassland",
    17: "Cold Temperate Evergreen Needle-leaf Forest",
    18: "Cold Temperate Deciduous Needle-leaf Forest",
    19: "Cold Temperate Open Woodland",
    20: "Cold Temperate Wooded Grassland",
    21: "Cold Temperate Grassland",
    22: "Polar Grassland",
    23: "Polar Open Woodland",
    24: "Polar Barren",
    25: "Warm Desert",
    26: "Cool Desert",
    27: "Polar Desert",
    28: "Mangrove",
    29: "Warm Freshwater Marsh",
    30: "Cool Freshwater Marsh",
    31: "Warm Saltwater Marsh",
    32: "Cool Saltwater Marsh",
    33: "Warm Peatland",
    34: "Cool Peatland",
    35: "Alpine Shrubland",
    36: "Alpine Grassland",
    37: "Urban",
    38: "Cropland",
    39: "Intensive Cropland",
    40: "Water",
    41: "Snow/Ice",
    42: "Bare Rock",
    43: "Bare Soil",
    44: "Sand",
    45: "Inland Water",
    46: "Ocean",
    47: "Lake"
}


def download_file(url, output_path, chunk_size=8192):
    """
    Download a file from URL with progress bar.
    
    Args:
        url (str): URL to download from
        output_path (str): Local path to save file
        chunk_size (int): Download chunk size in bytes
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Check if file already exists
        if Path(output_path).exists():
            print(f"File already exists: {output_path}")
            return True
        
        print(f"Downloading: {url}")
        print(f"Saving to: {output_path}")
        
        # Send GET request with stream
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get total file size
        total_size = int(response.headers.get('content-length', 0))
        
        # Download with progress bar
        with open(output_path, 'wb') as f, tqdm(
            desc=Path(output_path).name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    size = f.write(chunk)
                    pbar.update(size)
        
        print(f"✓ Download completed: {output_path}")
        return True
        
    except requests.RequestException as e:
        print(f"✗ Download error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def extract_archive(archive_path, extract_dir):
    """
    Extract compressed archive (zip, tar.gz) to directory.
    
    Args:
        archive_path (str): Path to archive file
        extract_dir (str): Directory to extract to
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        archive_path = Path(archive_path)
        extract_dir = Path(extract_dir)
        
        if not archive_path.exists():
            print(f"Archive not found: {archive_path}")
            return False
        
        # Create extraction directory
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Extracting: {archive_path}")
        print(f"To: {extract_dir}")
        
        # Handle different archive types
        if archive_path.suffix.lower() == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
                
        elif archive_path.suffixes[-2:] == ['.tar', '.gz']:
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_dir)
                
        elif archive_path.suffix.lower() == '.tar':
            with tarfile.open(archive_path, 'r') as tar_ref:
                tar_ref.extractall(extract_dir)
                
        else:
            print(f"Unsupported archive format: {archive_path}")
            return False
        
        print(f"✓ Extraction completed")
        
        # List extracted files
        extracted_files = list(extract_dir.rglob("*"))
        data_files = [f for f in extracted_files if f.is_file() and f.suffix in ['.nc', '.tif', '.asc', '.dat']]
        
        if data_files:
            print(f"Extracted {len(data_files)} data files:")
            for file in sorted(data_files):
                size = file.stat().st_size
                print(f"  - {file.name} ({size:,} bytes)")
        
        return True
        
    except Exception as e:
        print(f"✗ Extraction error: {e}")
        return False


def download_synmap_data(dataset="synmap_geotiff", output_dir="../../landcover/synmap"):
    """
    Download SYNMAP landcover dataset.
    
    Args:
        dataset (str): Dataset to download (synmap_global, synmap_ascii, synmap_geotiff)
        output_dir (str): Directory to save files
        
    Returns:
        bool: True if successful, False otherwise
    """
    if dataset not in SYNMAP_DATA:
        print(f"Unknown dataset: {dataset}")
        print(f"Available datasets: {list(SYNMAP_DATA.keys())}")
        return False
    
    data_info = SYNMAP_DATA[dataset]
    output_path = Path(output_dir) / data_info["filename"]
    
    print(f"=== Downloading SYNMAP Dataset ===")
    print(f"Dataset: {dataset}")
    print(f"Description: {data_info['description']}")
    print()
    
    # Download the file
    success = download_file(data_info["url"], str(output_path))
    
    if success:
        # Extract if it's an archive
        if output_path.suffix.lower() in ['.zip', '.tar', '.gz'] or output_path.suffixes[-2:] == ['.tar', '.gz']:
            extract_dir = output_path.parent / output_path.stem
            extract_success = extract_archive(str(output_path), str(extract_dir))
            
            if extract_success:
                print(f"\n✓ Dataset ready at: {extract_dir}")
            else:
                print(f"\n⚠ Downloaded but extraction failed: {output_path}")
        else:
            print(f"\n✓ Dataset ready at: {output_path}")
    
    return success


def visualize_synmap_data(data_path):
    """
    Visualize SYNMAP landcover data.
    
    Args:
        data_path (str): Path to SYNMAP data file (.nc, .tif, etc.)
    """
    try:
        data_path = Path(data_path)
        
        if not data_path.exists():
            print(f"Data file not found: {data_path}")
            return
        
        print(f"Visualizing: {data_path}")
        
        # Handle different file formats
        if data_path.suffix.lower() == '.nc':
            visualize_netcdf(data_path)
        elif data_path.suffix.lower() in ['.tif', '.tiff']:
            visualize_geotiff(data_path)
        else:
            print(f"Visualization not implemented for format: {data_path.suffix}")
            
    except Exception as e:
        print(f"Visualization error: {e}")


def visualize_netcdf(nc_path):
    """Visualize NetCDF SYNMAP data."""
    try:
        if xr is None:
            print("xarray not available. Install with: pip install xarray")
            return
        
        # Load the dataset
        ds = xr.open_dataset(nc_path)
        print("Dataset info:")
        print(ds)
        
        # Create multiple visualizations
        create_dominant_biome_map(ds)
        create_biome_fraction_maps(ds)
        create_biome_summary_stats(ds)
        
    except Exception as e:
        print(f"NetCDF visualization error: {e}")


def create_dominant_biome_map(ds):
    """Create a map showing dominant biome types."""
    try:
        
        # Get dominant biome type at each grid cell
        biome_type = ds['biome_type']
        
        # Create custom colormap for biome types
        fig, ax = plt.subplots(figsize=(16, 10), 
                              subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Create discrete colormap
        n_biomes = len(BIOME_MAPPING)
        colors = plt.cm.tab20(np.linspace(0, 1, n_biomes))
        cmap = mcolors.ListedColormap(colors)
        
        # Plot dominant biome types
        im = biome_type.plot(ax=ax,
                           transform=ccrs.PlateCarree(),
                           cmap=cmap,
                           add_colorbar=False,
                           vmin=0, vmax=47)
        
        # Add colorbar with biome names
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', 
                           shrink=0.8, aspect=30, pad=0.02)
        cbar.set_label('Dominant Biome Type', fontsize=12, rotation=270, labelpad=20)
        
        # Get unique biome IDs present in the data
        unique_biomes = np.unique(biome_type.values)
        unique_biomes = unique_biomes[unique_biomes >= 0]  # Remove any invalid values
        
        # Set custom tick positions and labels for biomes present in data
        if len(unique_biomes) <= 15:  # Show all if not too many
            tick_positions = unique_biomes
            tick_labels = [f"{biome_id}: {BIOME_MAPPING.get(biome_id, f'Biome {biome_id}')[:25]}" 
                          for biome_id in unique_biomes]
        else:  # Show subset if too many
            # Show every 2nd or 3rd biome to avoid overcrowding
            step = max(1, len(unique_biomes) // 10)
            tick_positions = unique_biomes[::step]
            tick_labels = [f"{biome_id}: {BIOME_MAPPING.get(biome_id, f'Biome {biome_id}')[:20]}" 
                          for biome_id in tick_positions]
        
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels(tick_labels, fontsize=8)
        
        # Add map features
        ax.coastlines(resolution='50m', color='black', linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.3)
        ax.gridlines(draw_labels=True, alpha=0.3)
        ax.set_global()
        
        plt.title('SYNMAP Global Dominant Biome Types', fontsize=16, pad=20)
        plt.tight_layout()
        
        # Save the figure
        output_path = Path("ELM-TAM/data/benchmark/landcover/synmap/synmap_dominant_biomes.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Dominant biome map saved to: {output_path.absolute()}")
        
        plt.show()
        
    except Exception as e:
        print(f"Dominant biome map error: {e}")


def create_biome_fraction_maps(ds, selected_biomes=None):
    """Create maps showing fractional coverage for selected biomes."""
    try:
        if selected_biomes is None:
            # Show most common biomes based on analysis
            selected_biomes = [0, 17, 38, 40, 41]  # Forest, Conifer, Cropland, Water, Snow/Ice
        
        n_maps = len(selected_biomes)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), 
                               subplot_kw={'projection': ccrs.PlateCarree()})
        axes = axes.flatten()
        
        for i, biome_id in enumerate(selected_biomes):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Get fractional coverage for this biome
            biome_frac = ds['biome_frac'].isel(type=biome_id)
            biome_name = BIOME_MAPPING.get(biome_id, f"Biome {biome_id}")
            
            # Plot fractional coverage
            im = biome_frac.plot(ax=ax,
                               transform=ccrs.PlateCarree(),
                               cmap='viridis',
                               add_colorbar=False,
                               vmin=0, vmax=1)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                              shrink=0.8, pad=0.05)
            cbar.set_label('Fractional Coverage', fontsize=10)
            
            # Add map features
            ax.coastlines(resolution='110m', color='white', linewidth=0.5)
            ax.set_global()
            ax.set_title(f'{biome_name}', fontsize=12, pad=10)
        
        # Hide unused subplots
        for i in range(len(selected_biomes), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('SYNMAP Biome Fractional Coverage', fontsize=16, y=0.98)
        plt.tight_layout()
        
        # Save the figure
        output_path = Path("ELM-TAM/data/benchmark/landcover/synmap/synmap_biome_fractions.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Biome fraction maps saved to: {output_path.absolute()}")
        
        plt.show()
        
    except Exception as e:
        print(f"Biome fraction maps error: {e}")


def create_biome_summary_stats(ds):
    """Create summary statistics for biome distribution."""
    try:
        
        print("\n=== SYNMAP Biome Summary Statistics ===")
        
        # Calculate global coverage for each biome
        biome_coverage = {}
        total_cells = ds.biome_frac.sizes['lat'] * ds.biome_frac.sizes['lon']
        
        for biome_id in range(48):
            # Sum fractional coverage across all grid cells
            total_frac = float(ds.biome_frac.isel(type=biome_id).sum())
            coverage_pct = (total_frac / total_cells) * 100
            
            if coverage_pct > 0.1:  # Only show biomes with >0.1% coverage
                biome_name = BIOME_MAPPING.get(biome_id, f"Biome {biome_id}")
                biome_coverage[biome_id] = {
                    'name': biome_name,
                    'coverage': coverage_pct
                }
        
        # Sort by coverage
        sorted_biomes = sorted(biome_coverage.items(), 
                             key=lambda x: x[1]['coverage'], reverse=True)
        
        print(f"\nTop biomes by global coverage:")
        print("-" * 60)
        for biome_id, info in sorted_biomes[:15]:  # Top 15
            print(f"{biome_id:2d}: {info['name']:<35} {info['coverage']:6.2f}%")
        
        # Dominant biome statistics
        unique_dominant, counts = np.unique(ds.biome_type.values, return_counts=True)
        
        print(f"\nDominant biome distribution (grid cells):")
        print("-" * 60)
        for biome_id, count in zip(unique_dominant, counts):
            pct = (count / total_cells) * 100
            biome_name = BIOME_MAPPING.get(biome_id, f"Biome {biome_id}")
            print(f"{biome_id:2d}: {biome_name:<35} {count:6d} ({pct:5.2f}%)")
        
    except Exception as e:
        print(f"Summary statistics error: {e}")


def visualize_geotiff(tif_path):
    """Visualize GeoTIFF SYNMAP data."""
    try:
        if rasterio is None:
            print("rasterio not available. Install with: pip install rasterio")
            return
        
        # Load the raster
        with rasterio.open(tif_path) as src:
            data = src.read(1)
            extent = [src.bounds.left, src.bounds.right, 
                     src.bounds.bottom, src.bounds.top]
        
        # Create map
        fig, ax = plt.subplots(figsize=(15, 10), 
                              subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Plot data
        im = ax.imshow(data, extent=extent, transform=ccrs.PlateCarree(),
                      cmap='tab20', alpha=0.9, vmin=0, vmax=47)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, orientation='horizontal', 
                    label='Land Cover Value', pad=0.05)
        
        # Add map features
        ax.coastlines(resolution='50m', color='black', linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5)
        ax.gridlines(draw_labels=True, alpha=0.5)
        
        plt.title(f'SYNMAP Land Cover: {tif_path.name}', fontsize=14, pad=20)
        
        # Save the figure
        output_path = tif_path.parent / f"{tif_path.stem}_visualization.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Map saved to: {output_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"GeoTIFF visualization error: {e}")


def main():
    """Main function with command line interface."""
    
    parser = argparse.ArgumentParser(description='Download and process SYNMAP landcover data')
    parser.add_argument('--dataset', choices=list(SYNMAP_DATA.keys()), 
                       default='synmap_geotiff',
                       help='SYNMAP dataset to download')
    parser.add_argument('--output_dir', '-o', type=str,
                       default='../../landcover/synmap',
                       help='Output directory for data files')
    parser.add_argument('--visualize', type=str,
                       help='Path to data file for visualization')
    parser.add_argument('--list-pfts', action='store_true',
                       help='List all plant functional types')
    
    args = parser.parse_args()
    
    if args.list_pfts:
        print("SYNMAP Biome Types:")
        for biome_id, biome_name in BIOME_MAPPING.items():
            print(f"  {biome_id:2d}: {biome_name}")
        return
    
    if args.visualize:
        visualize_synmap_data(args.visualize)
        return
    
    # Download dataset
    print("=== SYNMAP Data Downloader ===")
    success = download_synmap_data(args.dataset, args.output_dir)
    
    if success:
        print(f"\n🎉 SYNMAP data download completed!")
        print(f"Data saved to: {Path(args.output_dir).absolute()}")
        print("\nNext steps:")
        print("1. Use --visualize to create maps")
        print("2. Use --list-pfts to see all biome types")
        print("3. Integrate data into your ecosystem models")
        print("4. Analyze biome distributions by region")
    else:
        print("\n❌ Download failed. Please check the URLs and try again.")


if __name__ == "__main__":
    main()
