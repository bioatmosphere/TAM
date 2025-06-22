"""
Download and process TerraClimate data from the University of Idaho.

This script provides functionality to:
1. Download TerraClimate NetCDF files for specified variables and years
2. Visualize downloaded data with publication-quality maps
3. Handle multiple climate variables (precipitation, temperature, etc.)

Data source: https://climate.northwestknowledge.net/TERRACLIMATE-DATA

Dependencies:
    pip install requests xarray matplotlib cartopy tqdm
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr


def download_terraclimate_data(variables, start_year, end_year, output_dir):
    """
    Download TerraClimate data files for specified variables and years.
    
    Args:
        variables (list): List of climate variables (e.g., ['ppt', 'tmax', 'tmin'])
        start_year (int): Starting year for download
        end_year (int): Ending year for download
        output_dir (str): Directory to save downloaded files
    
    Returns:
        dict: Summary of download results
    """
    base_url = "https://climate.northwestknowledge.net/TERRACLIMATE-DATA"
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    results = {
        'successful': [],
        'failed': [],
        'skipped': []
    }
    
    total_files = len(variables) * (end_year - start_year + 1)
    print(f"Downloading {total_files} TerraClimate files...")
    
    with tqdm(total=total_files, desc="Overall Progress") as pbar:
        for variable in variables:
            for year in range(start_year, end_year + 1):
                filename = f"TerraClimate_{variable}_{year}.nc"
                file_path = Path(output_dir) / filename
                url = f"{base_url}/{filename}"
                
                # Skip if file already exists
                if file_path.exists():
                    file_size = file_path.stat().st_size
                    if file_size > 1024:  # Skip if file is larger than 1KB
                        print(f"Skipping {filename} (already exists)")
                        results['skipped'].append(filename)
                        pbar.update(1)
                        continue
                
                # Download file
                success = download_file(url, file_path, variable, year)
                if success:
                    results['successful'].append(filename)
                else:
                    results['failed'].append(filename)
                
                pbar.update(1)
                time.sleep(0.1)  # Be respectful to the server
    
    # Print summary
    print(f"\nDownload Summary:")
    print(f"  Successful: {len(results['successful'])}")
    print(f"  Failed: {len(results['failed'])}")
    print(f"  Skipped: {len(results['skipped'])}")
    
    if results['failed']:
        print(f"  Failed files: {', '.join(results['failed'])}")
    
    return results


def download_file(url, file_path, variable, year):
    """
    Download a single file with error handling and progress tracking.
    
    Args:
        url (str): URL to download from
        file_path (Path): Local path to save file
        variable (str): Climate variable name
        year (int): Year being downloaded
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"Downloading {variable} data for {year}...")
        
        # Make request with streaming and timeout
        response = requests.get(url, stream=True, timeout=(10, 60))
        response.raise_for_status()
        
        # Get file size for progress bar
        total_size = int(response.headers.get('content-length', 0))
        
        # Download with progress bar
        with open(file_path, 'wb') as f:
            if total_size > 0:
                with tqdm(total=total_size, unit='B', unit_scale=True, 
                         desc=f"{variable}_{year}", leave=False) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            else:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        
        # Verify download
        if file_path.exists() and file_path.stat().st_size > 0:
            print(f"✓ Downloaded {file_path.name} ({file_path.stat().st_size:,} bytes)")
            return True
        else:
            print(f"✗ Download failed: {file_path.name}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Download error for {variable}_{year}: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error for {variable}_{year}: {e}")
        return False


def visualize_terraclimate_data(data_dir, variable=None, year=None):
    """
    Create publication-quality visualizations of TerraClimate data.
    
    Args:
        data_dir (str): Directory containing TerraClimate NetCDF files
        variable (str, optional): Specific variable to plot
        year (int, optional): Specific year to plot
    """
    data_path = Path(data_dir)
    
    # Find available files
    if variable and year:
        nc_files = list(data_path.glob(f"TerraClimate_{variable}_{year}.nc"))
    elif variable:
        nc_files = list(data_path.glob(f"TerraClimate_{variable}_*.nc"))
    else:
        nc_files = list(data_path.glob("TerraClimate_*.nc"))
    
    if not nc_files:
        print(f"No TerraClimate files found in {data_dir}")
        return
    
    print(f"Found {len(nc_files)} files to visualize")
    
    # Create figures directory
    figures_dir = data_path / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    # Process each file
    for nc_file in nc_files:
        try:
            # Parse filename
            filename_parts = nc_file.stem.split('_')
            if len(filename_parts) >= 3:
                var_name = filename_parts[1]
                file_year = filename_parts[2]
            else:
                print(f"Cannot parse filename: {nc_file.name}")
                continue
            
            print(f"Visualizing {var_name} data for {file_year}...")
            
            # Load data
            with xr.open_dataset(nc_file) as ds:
                print(f"Dataset variables: {list(ds.data_vars)}")
                
                # Get the main data variable (usually the same as var_name)
                data_var = var_name if var_name in ds.data_vars else list(ds.data_vars)[0]
                data = ds[data_var]
                
                # Calculate annual mean if time dimension exists
                if 'time' in data.dims:
                    annual_mean = data.mean(dim='time')
                else:
                    annual_mean = data
                
                # Create visualization
                create_climate_map(annual_mean, var_name, file_year, figures_dir)
                
        except Exception as e:
            print(f"Error processing {nc_file.name}: {e}")


def create_climate_map(data, variable, year, output_dir):
    """
    Create a publication-quality map of climate data.
    
    Args:
        data (xarray.DataArray): Climate data to plot
        variable (str): Variable name
        year (str): Year
        output_dir (Path): Directory to save figure
    """
    # Variable-specific settings
    var_config = {
        'ppt': {
            'title': 'Annual Precipitation',
            'cmap': 'Blues',
            'units': 'mm/year'
        },
        'tmax': {
            'title': 'Annual Maximum Temperature',
            'cmap': 'Reds',
            'units': '°C'
        },
        'tmin': {
            'title': 'Annual Minimum Temperature',
            'cmap': 'coolwarm',
            'units': '°C'
        }
    }
    
    config = var_config.get(variable, {
        'title': f'{variable.upper()} Data',
        'cmap': 'viridis',
        'units': 'units'
    })
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 10), 
                          subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Add map features
    ax.coastlines(resolution='50m', color='black', linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5)
    ax.add_feature(cfeature.LAKES, alpha=0.3)
    ax.add_feature(cfeature.RIVERS, alpha=0.3)
    ax.gridlines(draw_labels=True, alpha=0.5)
    
    # Plot data
    if variable == 'tmin' or variable == 'tmax':
        # Convert Kelvin to Celsius for temperature
        plot_data = data - 273.15 if data.max() > 200 else data
    else:
        plot_data = data
    
    # Use percentiles for better color scaling
    vmin, vmax = np.percentile(plot_data.values[~np.isnan(plot_data.values)], [2, 98])
    
    im = plot_data.plot(ax=ax,
                       cmap=config['cmap'],
                       transform=ccrs.PlateCarree(),
                       vmin=vmin,
                       vmax=vmax,
                       add_colorbar=False)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                       label=f"{config['title']} ({config['units']})",
                       pad=0.05, shrink=0.8)
    
    # Set title
    plt.title(f"{config['title']} - {year}", fontsize=14, pad=20)
    
    # Save figure
    output_file = output_dir / f"{variable}_{year}_map.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Map saved to: {output_file}")
    
    plt.close()


def main():
    """
    Main function with command line interface for TerraClimate data processing.
    """
    parser = argparse.ArgumentParser(description='Download and process TerraClimate data')
    parser.add_argument('--mode', choices=['download', 'visualize'], default='download',
                       help='Mode: download data or visualize existing data')
    parser.add_argument('--variables', nargs='+', default=['ppt', 'tmax', 'tmin'],
                       help='Climate variables to download (e.g., ppt tmax tmin)')
    parser.add_argument('--start-year', type=int, default=2001,
                       help='Starting year for download')
    parser.add_argument('--end-year', type=int, default=2010,
                       help='Ending year for download')
    parser.add_argument('--output-dir', '-o', type=str,
                       default='../../ancillary/terraclimate',
                       help='Output directory for downloaded files')
    parser.add_argument('--variable', type=str,
                       help='Specific variable to visualize')
    parser.add_argument('--year', type=int,
                       help='Specific year to visualize')
    
    args = parser.parse_args()
    
    print("=== TerraClimate Data Processor ===")
    print(f"Mode: {args.mode}")
    print()
    
    if args.mode == 'download':
        # Download TerraClimate data
        results = download_terraclimate_data(
            args.variables,
            args.start_year,
            args.end_year,
            args.output_dir
        )
        
        if results['successful']:
            print(f"\n✓ Successfully downloaded {len(results['successful'])} files to: {args.output_dir}")
            print("Next steps:")
            print("1. Run with --mode visualize to create plots")
        else:
            print("✗ No files were successfully downloaded")
    
    elif args.mode == 'visualize':
        # Visualize existing data
        print(f"Visualizing TerraClimate data from: {args.output_dir}")
        visualize_terraclimate_data(args.output_dir, args.variable, args.year)
    
    print("\n=== Complete ===")


if __name__ == "__main__":
    main()