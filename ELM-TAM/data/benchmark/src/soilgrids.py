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

def extract_soil_data_for_points(lat_lon_file, output_file="soil_carbon_points.csv"):
    """
    Extract soil carbon data for specific coordinate points using SoilGrids REST API.
    
    Args:
        lat_lon_file (str): Path to CSV file with 'lat' and 'lon' columns
        output_file (str): Output CSV file for soil data
    """
    import pandas as pd
    import requests
    import time
    
    # Read coordinate points
    points_df = pd.read_csv(lat_lon_file)
    
    # Initialize results list
    results = []
    
    # SoilGrids REST API base URL
    base_url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
    
    print(f"Processing {len(points_df)} coordinate points...")
    
    for idx, row in points_df.iterrows():
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
            # Make API request with timeout and retry logic for rate limiting
            max_retries = 3
            retry_delay = 2
            
            for attempt in range(max_retries):
                response = requests.get(base_url, params=params, timeout=30)
                
                if response.status_code == 429:  # Rate limited
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    print(f"Rate limited, waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    break
            
            if response.status_code == 200:
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
            else:
                print(f"API error for point ({lat}, {lon}): {response.status_code}")
                results.append({'lat': lat, 'lon': lon, 'soil_carbon_stock': None})
                
        except Exception as e:
            print(f"Error processing point ({lat}, {lon}): {e}")
            results.append({'lat': lat, 'lon': lon, 'soil_carbon_stock': None})
        
        # Add delay to respect API rate limits
        time.sleep(1.0)  # Increased to 1 second between requests
        
        # Progress indicator
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(points_df)} points")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Soil carbon data saved to: {output_file}")
    
    return results_df


import rasterio
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import os
from pathlib import Path

def plot_soil_carbon(tiff_path="../ancillary/soilgrids/soil_carbon_stock.tif"):
    """
    Visualize soil organic carbon stock data with enhanced mapping features.
    """
    try:
        with rasterio.open(tiff_path) as src:
            data = src.read(1)
            
            # Create figure with map projection
            fig, ax = plt.subplots(figsize=(15, 10),
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
            cbar = plt.colorbar(img, 
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
    parser.add_argument('--mode', choices=['global', 'points'], default='points',
                       help='Download global data or extract for specific points')
    parser.add_argument('--lat-lon-file', type=str, 
                       default='../productivity/earth/lat_lon.csv',
                       help='CSV file with lat/lon coordinates for point extraction')
    parser.add_argument('--output', '-o', type=str, 
                       default='../ancillary/soilgrids/soil_carbon_points.csv',
                       help='Output file path')
    parser.add_argument('--output-dir', type=str, default='../ancillary/soilgrids',
                       help='Output directory for global download')
    
    args = parser.parse_args()
    
    if args.mode == 'points':
        # Extract data for specific points
        extract_soil_data_for_points(args.lat_lon_file, args.output)
    else:
        # Download global data
        download_soilgrids_data(args.output_dir)