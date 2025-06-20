"""
Download and process soil organic carbon stock data from SoilGrids.
Uses GDAL to handle remote VRT files and create optimized local GeoTIFFs.

https://www.isric.org/explore/soilgrids/soilgrids-access


https://www.isric.org/explore/soilgrids/faq-soilgrids

"""

from osgeo import gdal
import os
from pathlib import Path

def download_soilgrids_data(output_dir="./data"):
    """
    Download and process SoilGrids data with optimized settings.
    
    Args:
        output_dir (str): Directory to save the processed GeoTIFF
    """
    try:
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Configure GDAL settings for optimized output and faster download
        # Set GDAL options for faster networking
        gdal.SetConfigOption('GDAL_HTTP_TIMEOUT', '300')
        gdal.SetConfigOption('GDAL_HTTP_LOW_SPEED_TIME', '30')
        gdal.SetConfigOption('GDAL_HTTP_LOW_SPEED_LIMIT', '1000')
        gdal.SetConfigOption('CPL_VSIL_CURL_CACHE_SIZE', '134217728')  # 128MB cache
        gdal.SetConfigOption('VSI_CACHE_SIZE', '268435456')  # 256MB cache
        
        kwargs = {
            'format': 'GTiff',
            'creationOptions': [
                "TILED=YES",          # Enable tiling for faster access
                "COMPRESS=DEFLATE",   # Use compression
                "PREDICTOR=2",        # Optimize compression for continuous data
                "BIGTIFF=YES",        # Support large files
                "NUM_THREADS=ALL_CPUS"  # Use all CPU cores
            ]
        }
        
        # Source VRT URL with optimized curl options
        src_url = ('/vsicurl?max_retry=5&retry_delay=0.5&list_dir=no&'
                   'tcp_keepalive=yes&timeout=300&low_speed_limit=1000&'
                   'low_speed_time=30&url='
                   'https://files.isric.org/soilgrids/latest/data/ocs/'
                   'ocs_0-30cm_mean.vrt')
        
        # Output file path
        output_file = os.path.join(output_dir, 'soil_carbon_stock.tif')
        
        print(f"Downloading soil carbon stock data to: {output_file}")
        
        # Download and process the data
        ds = gdal.Translate(output_file, src_url, **kwargs)
        
        if ds is not None:
            print("Download completed successfully")
            del ds  # Close the dataset
            return True
            
    except Exception as e:
        print(f"Error downloading SoilGrids data: {e}")
        return False

def extract_soil_data_for_points(lat_lon_file, output_file="soil_carbon_points.csv", 
                                checkpoint_file=None, batch_size=50):
    """
    Extract soil carbon data for specific coordinate points using SoilGrids REST API.
    
    Args:
        lat_lon_file (str): Path to CSV file with 'lat' and 'lon' columns
        output_file (str): Output CSV file for soil data
        checkpoint_file (str): File to save progress (auto-generated if None)
        batch_size (int): Number of points to process before saving checkpoint
    """
    import pandas as pd
    import requests
    import time
    from pathlib import Path
    
    # Read coordinate points
    points_df = pd.read_csv(lat_lon_file)
    
    # Setup checkpoint file
    if checkpoint_file is None:
        checkpoint_file = output_file.replace('.csv', '_checkpoint.csv')
    
    # Check if we have existing progress
    start_index = 0
    results = []
    
    if Path(checkpoint_file).exists():
        print(f"Found checkpoint file: {checkpoint_file}")
        existing_df = pd.read_csv(checkpoint_file)
        results = existing_df.to_dict('records')
        start_index = len(results)
        print(f"Resuming from point {start_index + 1}/{len(points_df)}")
    
    # SoilGrids REST API base URL
    base_url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
    
    print(f"Processing {len(points_df)} coordinate points (starting from {start_index + 1})...")
    
    for idx, row in points_df.iterrows():
        # Skip if we've already processed this point
        if idx < start_index:
            continue
            
        lat, lon = row['lat'], row['lon']
        
        # API parameters for soil organic carbon stock (0-30cm)
        params = {
            'lon': lon,
            'lat': lat,
            'property': 'ocs',
            'depth': '0-30cm',
            'value': 'mean'
        }
        
        try:
            # Make API request with timeout and retry logic
            max_retries = 3
            retry_delay = 2
            response = None
            
            for attempt in range(max_retries):
                try:
                    response = requests.get(base_url, params=params, 
                                          timeout=(10, 60))  # (connect, read) timeout
                    
                    if response.status_code == 429:  # Rate limited
                        wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                        print(f"Rate limited, waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        break
                        
                except requests.exceptions.Timeout:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        print(f"Timeout for point ({lat}, {lon}), retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"Final timeout for point ({lat}, {lon}), skipping...")
                        results.append({'lat': lat, 'lon': lon, 'soil_carbon_stock': None})
                        break
            
            if response and response.status_code == 200:
                data = response.json()
                
                # Extract soil carbon value
                if 'properties' in data and 'layers' in data['properties']:
                    layers = data['properties']['layers']
                    if layers and 'depths' in layers[0]:
                        depths = layers[0]['depths']
                        if depths and 'values' in depths[0]:
                            soil_carbon = depths[0]['values']['mean']
                            
                            results.append({
                                'lat': lat,
                                'lon': lon,
                                'soil_carbon_stock': soil_carbon
                            })
                        else:
                            results.append({'lat': lat, 'lon': lon, 'soil_carbon_stock': None})
                else:
                    results.append({'lat': lat, 'lon': lon, 'soil_carbon_stock': None})
            elif response:
                print(f"API error for point ({lat}, {lon}): {response.status_code}")
                results.append({'lat': lat, 'lon': lon, 'soil_carbon_stock': None})
            # If response is None (timeout handled above), skip this section
                
        except Exception as e:
            print(f"Error processing point ({lat}, {lon}): {e}")
            results.append({'lat': lat, 'lon': lon, 'soil_carbon_stock': None})
        
        # Add delay to respect API rate limits
        time.sleep(1.0)  # Increased to 1 second between requests
        
        # Save checkpoint every batch_size points
        if (idx + 1) % batch_size == 0:
            checkpoint_df = pd.DataFrame(results)
            checkpoint_df.to_csv(checkpoint_file, index=False)
            print(f"Processed {idx + 1}/{len(points_df)} points - checkpoint saved")
        elif (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(points_df)} points")
    
    # Save final results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Soil carbon data saved to: {output_file}")
    
    # Clean up checkpoint file
    if Path(checkpoint_file).exists():
        Path(checkpoint_file).unlink()
        print(f"Checkpoint file removed: {checkpoint_file}")
    
    return results_df


def plot_soil_carbon_points(csv_file, output_dir="../ancillary/soilgrids/figures"):
    """
    Create publication-quality plots of soil carbon data points.
    
    Args:
        csv_file (str): Path to CSV file with soil carbon data
        output_dir (str): Directory to save the plots
    
    Returns:
        pd.DataFrame: Cleaned dataset with valid soil carbon values
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import numpy as np
    from pathlib import Path
    import matplotlib.colors as mcolors
    
    # Set publication-quality style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'figure.titlesize': 16
    })
    
    # Read and validate data
    try:
        df = pd.read_csv(csv_file)
        required_cols = ['lat', 'lon', 'soil_carbon_stock']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV must contain columns: {required_cols}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None
    
    # Clean data and calculate statistics
    df_clean = df.dropna(subset=['soil_carbon_stock'])
    df_missing = df[df['soil_carbon_stock'].isna()]
    
    if len(df_clean) == 0:
        print("No valid soil carbon data found!")
        return None
    
    # Calculate comprehensive statistics
    stats = {
        'count': len(df_clean),
        'missing': len(df_missing),
        'mean': df_clean['soil_carbon_stock'].mean(),
        'median': df_clean['soil_carbon_stock'].median(),
        'std': df_clean['soil_carbon_stock'].std(),
        'min': df_clean['soil_carbon_stock'].min(),
        'max': df_clean['soil_carbon_stock'].max(),
        'q25': df_clean['soil_carbon_stock'].quantile(0.25),
        'q75': df_clean['soil_carbon_stock'].quantile(0.75)
    }
    
    print(f"Dataset Summary:")
    print(f"  Valid points: {stats['count']}")
    print(f"  Missing data: {stats['missing']}")
    print(f"  Range: {stats['min']:.1f} - {stats['max']:.1f} t C/ha")
    print(f"  Mean ± SD: {stats['mean']:.1f} ± {stats['std']:.1f} t C/ha")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # === WORLD MAP ===
    fig, ax = plt.subplots(figsize=(16, 10), 
                          subplot_kw={'projection': ccrs.Robinson()})
    
    # Enhanced map features
    ax.add_feature(cfeature.LAND, facecolor='#f5f5f5', alpha=0.7)
    ax.add_feature(cfeature.OCEAN, facecolor='#e6f3ff', alpha=0.7)
    ax.add_feature(cfeature.COASTLINE, color='#333333', linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, color='#666666', linewidth=0.4, alpha=0.6)
    ax.add_feature(cfeature.LAKES, facecolor='#cce7ff', alpha=0.5)
    ax.add_feature(cfeature.RIVERS, color='#66b3ff', alpha=0.3)
    
    # Set global extent with slight margin
    ax.set_global()
    
    # Enhanced gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='#999999', 
                     alpha=0.7, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    
    # Create custom colormap with better contrast
    colors = ['#ffffcc', '#fed976', '#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c', '#bd0026', '#800026']
    n_bins = 100
    cmap = mcolors.LinearSegmentedColormap.from_list('soil_carbon', colors, N=n_bins)
    
    # Create scatter plot with improved styling
    vmin, vmax = np.percentile(df_clean['soil_carbon_stock'], [5, 95])  # Use 5-95 percentile for better contrast
    
    scatter = ax.scatter(df_clean['lon'], df_clean['lat'],
                        c=df_clean['soil_carbon_stock'],
                        cmap=cmap,
                        s=50,
                        alpha=0.8,
                        transform=ccrs.PlateCarree(),
                        edgecolors='black',
                        linewidth=0.5,
                        vmin=vmin,
                        vmax=vmax,
                        zorder=5)
    
    # Plot missing data points in grey
    if len(df_missing) > 0:
        ax.scatter(df_missing['lon'], df_missing['lat'],
                  c='lightgray',
                  s=30,
                  alpha=0.6,
                  transform=ccrs.PlateCarree(),
                  edgecolors='darkgray',
                  linewidth=0.3,
                  label=f'Missing data (n={len(df_missing)})',
                  zorder=4)
        ax.legend(loc='lower left', framealpha=0.9)
    
    # Enhanced colorbar
    cbar = plt.colorbar(scatter, ax=ax, 
                       orientation='horizontal',
                       label='Soil Organic Carbon Stock (t C/ha)',
                       pad=0.08,
                       shrink=0.8,
                       aspect=40)
    cbar.ax.tick_params(labelsize=10)
    
    # Professional title with subtitle
    plt.suptitle('Global Distribution of Soil Organic Carbon Stock', 
                fontsize=16, fontweight='bold', y=0.95)
    ax.set_title(f'Study Sites with Soil Carbon Measurements (n={stats["count"]})', 
                fontsize=12, pad=15)
    
    # Save map
    map_path = Path(output_dir) / 'soil_carbon_global_map.pdf'
    plt.savefig(map_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    print(f"World map saved to: {map_path}")
    
    # === ENHANCED HISTOGRAM WITH STATISTICS ===
    fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Histogram with KDE overlay
    n_bins = min(30, len(df_clean) // 5)  # Adaptive bin number
    ax2.hist(df_clean['soil_carbon_stock'], 
             bins=n_bins, 
             alpha=0.7, 
             color='#8b4513',
             edgecolor='black',
             linewidth=0.7,
             density=True)
    
    # Add KDE curve
    from scipy import stats
    kde = stats.gaussian_kde(df_clean['soil_carbon_stock'])
    x_range = np.linspace(stats['min'], stats['max'], 100)
    ax2.plot(x_range, kde(x_range), 'r-', linewidth=2, label='Density curve')
    
    # Add statistical markers
    ax2.axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label=f'Mean ({stats["mean"]:.1f})')
    ax2.axvline(stats['median'], color='blue', linestyle='--', linewidth=2, label=f'Median ({stats["median"]:.1f})')
    
    ax2.set_xlabel('Soil Organic Carbon Stock (t C/ha)')
    ax2.set_ylabel('Density')
    ax2.set_title('Distribution of Soil Carbon Values')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Box plot with detailed statistics
    ax3.boxplot(df_clean['soil_carbon_stock'], 
               patch_artist=True,
               boxprops=dict(facecolor='#deb887', alpha=0.7),
               medianprops=dict(color='red', linewidth=2))
    
    ax3.set_ylabel('Soil Organic Carbon Stock (t C/ha)')
    ax3.set_title('Statistical Summary')
    ax3.grid(True, alpha=0.3)
    
    # Add detailed statistics text
    stats_text = f"""Dataset Statistics (n={stats['count']})
    
Range: {stats['min']:.1f} - {stats['max']:.1f} t C/ha
Mean: {stats['mean']:.1f} t C/ha
Median: {stats['median']:.1f} t C/ha
Std Dev: {stats['std']:.1f} t C/ha

Quartiles:
Q1 (25%): {stats['q25']:.1f} t C/ha
Q3 (75%): {stats['q75']:.1f} t C/ha
IQR: {stats['q75'] - stats['q25']:.1f} t C/ha

Data Quality:
Valid: {stats['count']} points
Missing: {stats['missing']} points
Coverage: {stats['count']/(stats['count']+stats['missing'])*100:.1f}%"""
    
    ax3.text(1.1, 0.5, stats_text, transform=ax3.transAxes, 
             fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save histogram
    hist_path = Path(output_dir) / 'soil_carbon_statistics.pdf'
    plt.savefig(hist_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    print(f"Statistics plot saved to: {hist_path}")
    
    # Display plots
    plt.show()
    
    # Save summary statistics to CSV
    stats_df = pd.DataFrame([stats])
    stats_path = Path(output_dir) / 'soil_carbon_summary_stats.csv'
    stats_df.to_csv(stats_path, index=False)
    print(f"Summary statistics saved to: {stats_path}")
    
    return df_clean

def plot_soil_carbon(tiff_path="../ancillary/soilgrids/soil_carbon_stock.tif"):
    """
    Visualize soil organic carbon stock data with enhanced mapping features.
    """
    import rasterio
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import numpy as np
    from pathlib import Path
    
    try:
        with rasterio.open(tiff_path) as src:
            data = src.read(1)
            
            # Create figure with map projection
            _, ax = plt.subplots(figsize=(15, 10),
                               subplot_kw={'projection': ccrs.Robinson()})
            
            # Add map features using cfeature
            ax.coastlines(resolution='50m', color='black', linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            ax.add_feature(cfeature.LAKES, alpha=0.5)
            ax.add_feature(cfeature.RIVERS, alpha=0.5)
            
            # Plot the data
            img = ax.imshow(data,
                          transform=ccrs.PlateCarree(),
                          cmap='YlOrBr',
                          vmin=0,
                          vmax=np.nanpercentile(data, 95))
            
            # Add colorbar
            plt.colorbar(img, 
                        orientation='horizontal',
                        label='Soil Organic Carbon Stock (t/ha)',
                        pad=0.05)
            
            plt.title('Global Soil Organic Carbon Stock (0-30cm)', pad=20)
            
            # Save figure
            output_dir = "../ancillary/soilgrids/figures"
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(f"{output_dir}/soil_carbon_map.pdf",
                       dpi=300,
                       bbox_inches='tight')
            
            print(f"Map saved to: {output_dir}/soil_carbon_map.pdf")
            plt.show()
            
    except Exception as e:
        print(f"Error visualizing data: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download SoilGrids data')
    parser.add_argument('--mode', choices=['global', 'points', 'plot'], default='points',
                       help='Download global data, extract for specific points, or plot existing data')
    parser.add_argument('--lat-lon-file', type=str, 
                       default='../productivity/earth/lat_lon.csv',
                       help='CSV file with lat/lon coordinates for point extraction')
    parser.add_argument('--output', '-o', type=str, 
                       default='../ancillary/soilgrids/soil_carbon_points.csv',
                       help='Output file path')
    parser.add_argument('--output-dir', type=str, default='../ancillary/soilgrids',
                       help='Output directory for global download')
    parser.add_argument('--checkpoint-file', type=str,
                       help='Custom checkpoint file path (auto-generated if not specified)')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='Number of points to process before saving checkpoint')
    parser.add_argument('--plot-file', type=str,
                       help='CSV file to plot (uses --output path if not specified)')
    
    args = parser.parse_args()
    
    if args.mode == 'points':
        # Extract data for specific points
        extract_soil_data_for_points(args.lat_lon_file, args.output, 
                                   args.checkpoint_file, args.batch_size)
    elif args.mode == 'plot':
        # Plot existing soil carbon data
        plot_file = args.plot_file if args.plot_file else args.output
        plot_soil_carbon_points(plot_file)
    else:
        # Download global data
        download_soilgrids_data(args.output_dir)