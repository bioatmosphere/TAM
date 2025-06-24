""" 
Aggregate processed data from different sources into a single dataset for machine learing.

This script fetches data from various sources, processes them, and aggregate them into a single DataFrame
    for machine learning (deep learning) next.

What it does:
    - Reads grassland and forest data from specified CSV files.
    - Renames columns to ensure consistency across datasets.
    - Combines the data into a single DataFrame.
    - Plots the spatial distribution of data points on a map.

**NOTE**: It extracts the latitude and longitude of data points to be used for extracting
    data from the ancillary data sources like weather, soil, and other environmental data.
    
The data sources include:
    - grassland data from Dryad
    - forest data from ForC
    - other potential sources in the future.
    - ancillary data from various sources.

Check out https://github.com/NVIDIA-Omniverse-blueprints/earth2-weather-analytics 
    for inspirations for structure and data sources.

MCP: https://claude.ai/share/1d62b6fc-271b-422f-8c0d-f9be2efdd8ed

"""

import pandas as pd
import os
from pathlib import Path
import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Import scipy for spatial distance calculations
try:
    from scipy.spatial.distance import cdist
    SCIPY_AVAILABLE = True
except ImportError:
    print("Warning: scipy not available. Spatial merging will use simplified approach.")
    SCIPY_AVAILABLE = False

def process_productivity_data(df):
    """Process grassland/forest productivity data with standard column renaming."""
    # Make column names consistent
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        df.rename(columns={'Latitude': 'lat', 'Longitude': 'lon'}, inplace=True)
    if 'mean' in df.columns:
        df.rename(columns={'mean': 'BNPP'}, inplace=True)
    if 'dominant.veg' in df.columns:
        df.rename(columns={'dominant.veg': 'biome'}, inplace=True)
    if 'Grassland type' in df.columns:
        df.rename(columns={'Grassland type': 'biome'}, inplace=True)
    
    return df

def process_terraclimate_data(df, file_path):
    """Process TerraClimate mean data with appropriate column naming."""
    # Ensure consistent lat/lon column names
    if 'latitude' in df.columns and 'longitude' in df.columns:
        df.rename(columns={'latitude': 'lat', 'longitude': 'lon'}, inplace=True)
    
    # Extract variable name from file path and rename mean_value column
    file_name = file_path.stem
    if '_means' in file_name:
        var_name = file_name.replace('_means', '')
        if 'mean_value' in df.columns:
            df.rename(columns={'mean_value': var_name}, inplace=True)
    
    # Add data source identifier
    df['data_source'] = 'terraclimate'
    
    return df

def merge_datasets_by_location(productivity_df, climate_df, tolerance=0.1):
    """
    Merge productivity and climate datasets based on lat/lon coordinates.
    
    Args:
        productivity_df: DataFrame with productivity data (must have 'lat', 'lon' columns)
        climate_df: DataFrame with climate data (must have 'lat', 'lon' columns)
        tolerance: Distance tolerance in degrees for matching coordinates
    
    Returns:
        Merged DataFrame with climate variables added to productivity data
    """
    # Extract coordinates
    prod_coords = productivity_df[['lat', 'lon']].values
    clim_coords = climate_df[['lat', 'lon']].values
    
    # Calculate distances between all productivity and climate points
    if SCIPY_AVAILABLE:
        distances = cdist(prod_coords, clim_coords)
    else:
        # Fallback: simple Euclidean distance calculation
        distances = np.sqrt(
            ((prod_coords[:, np.newaxis, 0] - clim_coords[np.newaxis, :, 0]) ** 2) +
            ((prod_coords[:, np.newaxis, 1] - clim_coords[np.newaxis, :, 1]) ** 2)
        )
    
    # Find nearest climate point for each productivity point
    nearest_indices = np.argmin(distances, axis=1)
    nearest_distances = np.min(distances, axis=1)
    
    # Create a copy of productivity data to avoid modifying original
    merged_df = productivity_df.copy()
    
    # Add climate variables for points within tolerance
    climate_vars = [col for col in climate_df.columns if col not in ['lat', 'lon', 'point_name', 'units', 'data_source']]
    
    for var in climate_vars:
        merged_df[var] = np.nan
    
    # Merge climate data where distance is within tolerance
    valid_matches = nearest_distances <= tolerance
    
    for i, (is_valid, clim_idx) in enumerate(zip(valid_matches, nearest_indices)):
        if is_valid:
            for var in climate_vars:
                if var in climate_df.columns:
                    merged_df.iloc[i, merged_df.columns.get_loc(var)] = climate_df.iloc[clim_idx][var]
    
    print(f"Merged {valid_matches.sum()} out of {len(productivity_df)} productivity points with climate data")
    print(f"Climate variables added: {climate_vars}")
    
    return merged_df

def aggregate_data(data_files=[]):
    """Aggregate data from different sources into a single DataFrame.
    
    Handles both productivity data (grassland/forest) and ancillary climate data (TerraClimate).
    """
    
    # Define base directories
    productivity_dir = Path('../productivity').resolve()
    base_dir = Path('..').resolve()
    
    # Initialize dictionaries to hold DataFrames by type
    productivity_dataframes = []
    climate_dataframes = []
    
    # Read each file and append the DataFrame to the appropriate list
    for file in data_files:
        try:
            # Determine file path based on file location
            if file.startswith('../ancillary/'):
                # TerraClimate ancillary data files
                file_path = base_dir / file.replace('../', '')
            else:
                # Productivity data files (grassland/forest)
                file_path = productivity_dir / file
                # Verify productivity directory exists
                if not productivity_dir.exists():
                    raise FileNotFoundError(f"Productivity data directory not found: {productivity_dir}")
            
            if not file_path.is_file():
                print(f"Warning: Data file not found: {file_path}, skipping...")
                continue
                
            print(f"Processing: {file_path}")
            df = pd.read_csv(file_path)
            print(f"Loaded {len(df)} rows from {file}")

            # Process based on file type
            if 'terraclimate' in str(file_path):
                # TerraClimate data processing
                df = process_terraclimate_data(df, file_path)
                climate_dataframes.append(df)
            else:
                # Productivity data processing (grassland/forest)
                df = process_productivity_data(df)
                productivity_dataframes.append(df)

        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue
    
    # Combine and merge the datasets
    integrated_df = None
    
    # First, concatenate productivity data
    if productivity_dataframes:
        productivity_df = pd.concat(productivity_dataframes, ignore_index=True)
        productivity_df['data_source'] = 'productivity'
        print(f"Combined productivity data: {len(productivity_df)} rows")
        integrated_df = productivity_df
    
    # Then, process climate data and merge with productivity data if possible  
    if climate_dataframes and integrated_df is not None:
        print("Merging climate variables with productivity data...")
        
        # Process each climate variable separately to avoid complex merges
        for df in climate_dataframes:
            # Get the variable name from the dataframe columns
            var_cols = [col for col in df.columns if col not in ['lat', 'lon', 'point_name', 'units', 'data_source']]
            if var_cols:
                var_name = var_cols[0]
                print(f"  Processing {var_name}...")
                
                # Simple approach: find exact coordinate matches
                matched_count = 0
                for idx, row in integrated_df.iterrows():
                    prod_lat, prod_lon = row['lat'], row['lon']
                    
                    # Find matching climate data point
                    climate_match = df[
                        (abs(df['lat'] - prod_lat) < 0.01) & 
                        (abs(df['lon'] - prod_lon) < 0.01)
                    ]
                    
                    if not climate_match.empty:
                        integrated_df.at[idx, var_name] = climate_match.iloc[0][var_name]
                        matched_count += 1
                    else:
                        integrated_df.at[idx, var_name] = np.nan
                
                print(f"    Matched {matched_count}/{len(integrated_df)} points for {var_name}")
    
    elif climate_dataframes and integrated_df is None:
        print("Only climate data available, no productivity data to merge with")
        integrated_df = pd.concat(climate_dataframes, ignore_index=True)
    
    if integrated_df is not None and len(integrated_df) > 0:
        # Filter to keep only essential columns: lat, lon, biome, productivity, and climate data
        essential_columns = ['lat', 'lon']
        
        # Add biome column (ecosystem type)
        biome_cols = [col for col in integrated_df.columns if col in ['biome', 'ecosystem', 'vegetation_type', 'land_cover']]
        if biome_cols:
            essential_columns.extend(biome_cols)
            print(f"Found biome column: {biome_cols}")
        else:
            print("Warning: No biome column found")
        
        # Add productivity column (BNPP or similar)
        productivity_cols = [col for col in integrated_df.columns if col in ['BNPP', 'mean', 'productivity']]
        if productivity_cols:
            essential_columns.extend(productivity_cols)
        else:
            print("Warning: No productivity column found. Looking for numeric columns...")
            # Fallback: look for numeric columns that might represent productivity
            numeric_cols = integrated_df.select_dtypes(include=[np.number]).columns
            potential_prod_cols = [col for col in numeric_cols if col not in ['lat', 'lon', 'aet', 'pet', 'ppt', 'tmax', 'tmin']]
            if potential_prod_cols:
                essential_columns.extend(potential_prod_cols[:1])  # Take first one
                print(f"Using {potential_prod_cols[0]} as productivity measure")
        
        # Add climate variables
        climate_cols = [col for col in integrated_df.columns if col in ['aet', 'pet', 'ppt', 'tmax', 'tmin']]
        essential_columns.extend(climate_cols)
        
        # Filter the dataframe to keep only essential columns
        available_cols = [col for col in essential_columns if col in integrated_df.columns]
        filtered_df = integrated_df[available_cols].copy()
        
        print(f"Filtered dataset from {integrated_df.shape[1]} to {filtered_df.shape[1]} columns")
        print(f"Essential columns: {available_cols}")
        
        # Write the filtered DataFrame to a CSV file
        output_file = productivity_dir / 'earth' / 'aggregated_data.csv'
        # check if the file exists, if not save it
        if not output_file.exists():
            # Ensure parent directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)
            filtered_df.to_csv(output_file, index=False)
            print(f"Aggregated data saved to: {output_file}")
        else:
            print(f"File already exists: {output_file}, skipping save.")
        
        # Write only the columns of lat and lon to a separate file
        # this file will be used to extract data from ancillary data sources
        lat_lon_file = productivity_dir / 'earth' / 'lat_lon.csv'
        if not lat_lon_file.exists():
            lat_lon_df = filtered_df[['lat', 'lon']].drop_duplicates()  # Remove duplicates
            lat_lon_df.to_csv(lat_lon_file, index=False)
            print(f"Latitude and longitude data saved to: {lat_lon_file}")
        else:
            print(f"File already exists: {lat_lon_file}, skipping save.")
        
        # Return the filtered DataFrame
        return filtered_df
    else:
        print("No data files were found or processed successfully.")
        return pd.DataFrame()  # Return an empty DataFrame if no files were found


def plot_data_distribution(df):
    """ Plot the spatial distribution of data points

    TODO:
        - Add more map features like lakes, rivers, etc.
        - Customize the legend to show unique PFTs with colors.
        - Add more descriptive titles and labels.

    Parameters:
        df: dataframe
            the dataframe to be plotted.
    """

    # Create a figure with a PlateCarree projection
    fig, ax = plt.subplots(figsize=(15, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Add map features
    ax.coastlines(resolution='50m', color='black', linewidth=0.5)
    # ax.add_feature(cfeature.BORDERS, linestyle=':')
    # ax.add_feature(cfeature.LAKES, alpha=0.5)
    # ax.add_feature(cfeature.LAND, facecolor='lightgray')
    # ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    
    # Set map extent [lon_min, lon_max, lat_min, lat_max]
    ax.set_extent([-180, 180, -90, 90])
    
    # Add gridlines with labels
    gl = ax.gridlines(draw_labels=True, 
                     linewidth=0.5, 
                     color='gray', 
                     alpha=0.5,
                     linestyle='--')
    gl.top_labels = False  # Remove top labels
    gl.right_labels = False  # Remove right labels
    
    
    # Create a color map for unique PFTs
    unique_cover = df['biome'].unique()
    #colors = plt.cm.Set3(np.linspace(0, 1, len(unique_pfts)))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_cover)))
    
    # Plot points for each PFT category
    for cover, color in zip(unique_cover, colors):
        #if cover == 'evergreen broadleaf forest':
            mask = df['biome'] == cover
            ax.scatter(df.loc[mask, 'lon'], 
                       df.loc[mask, 'lat'],
                       color=color,
                       s=50,
                       label=cover,
                       transform=ccrs.PlateCarree()
                      )
    
    # Adjust legend position and size
    # ax.legend(title='Type',
    #          bbox_to_anchor=(0.0, 0.1),
    #          loc='lower left',
    #          shadow=True,                # Add shadow
    #          borderpad=1                 # Padding between legend and frame
    #          )
    
    # Set title
    ax.set_title(f'Data ({df.shape[0]}) with BNPP Measurements', fontsize=15)
    
    # Save the figure as a PDF with high resolution
    output_dir = '../productivity/earth'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'aggregated_data_distribution.pdf')
    plt.savefig(output_path, 
                format='pdf',
                dpi=300,              # High resolution
                bbox_inches='tight',  # Prevent cutoff
                facecolor='white',    # White background
                edgecolor='none',     # No edge color
                pad_inches=0.1)       # Add small padding
    print(f"\nMap saved as: {output_path}")
    
    # Display the plot
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Aggregate data from different sources for machine learning')
    parser.add_argument('--files', '-f', nargs='+', 
                       default=['grassland/grassland_component_df.csv',
                                'forc/forest_component_df.csv',
                                '../ancillary/terraclimate/point_extractions/aet_means.csv',
                                '../ancillary/terraclimate/point_extractions/pet_means.csv',
                                '../ancillary/terraclimate/point_extractions/ppt_means.csv',
                                '../ancillary/terraclimate/point_extractions/tmax_means.csv',
                                '../ancillary/terraclimate/point_extractions/tmin_means.csv'],
                       help='List of data files to aggregate (default: grassland, forest, and terraclimate mean files)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plotting the data distribution')
    parser.add_argument('--output-dir', '-o', type=str, default='../productivity',
                       help='Output directory for processed data (default: ../productivity)')
    
    args = parser.parse_args()
    
    try:
        # Aggregate data from the specified files
        integrated_data = aggregate_data(args.files)
        # Check if the integrated DataFrame is not empty and plot the data distribution
        if not integrated_data.empty:
            print(f"Integrated DataFrame shape: {integrated_data.shape}")
            print(integrated_data.head())
            # Plot the integrated data if needed
            if not args.no_plot:
                plot_data_distribution(integrated_data)
        else:
            print("No data to display.")
    except Exception as e:
        print(f"An error occurred during data integration: {str(e)}")


if __name__ == "__main__":
    main()
