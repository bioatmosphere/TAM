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

def aggregate_data(data_files=[]):
    """Aggregate data from different sources into a single DataFrame.
    
    """
    
    # Define and validate the data directory path
    data_dir = Path('../productivity').resolve()
    # Verify the directory exists
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    print(f"Using data directory: {data_dir}")
    
    # Initialize an empty list to hold DataFrames
    dataframes = []
    
    # Read each file and append the DataFrame to the list
    for file in data_files:
        try:
            file_path = data_dir / file
            if not data_dir.exists():
                raise FileNotFoundError(f"Data directory not found: {data_dir}")
            if not file_path.is_file():
                raise FileNotFoundError(f"Data file not found: {file_path}")
            print(f"Processing: {file_path}")
            df = pd.read_csv(file_path)
            print(f"Loaded {len(df)} rows from {file}")

            # make column names consistent
            if 'Latitude' in df.columns and 'Longitude' in df.columns:
                df.rename(columns={'Latitude': 'lat', 'Longitude': 'lon'}, inplace=True)
            if 'mean' in df.columns:
                df.rename(columns={'mean': 'BNPP'}, inplace=True)
            if 'dominant.veg' in df.columns:
                df.rename(columns={'dominant.veg': 'biome'}, inplace=True)
            if 'Grassland type' in df.columns:
                df.rename(columns={'Grassland type': 'biome'}, inplace=True)

            # append the DataFrame to the list
            dataframes.append(df)

        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue
    
    # Concatenate all DataFrames into a single DataFrame
    if dataframes:
        integrated_df = pd.concat(dataframes, ignore_index=True)
        # Write the integrated DataFrame to a CSV file
        output_file = data_dir / 'earth' / 'aggregated_data.csv'
        # check if the file exists, if not save it
        if not output_file.exists():
            # Ensure parent directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)
            integrated_df.to_csv(output_file, index=False)
            print(f"Aggregated data saved to: {output_file}")
        else:
            print(f"File already exists: {output_file}, skipping save.")
        # Write only the columns of lat and lon to a separate file
        lat_lon_file = data_dir / 'earth' / 'lat_lon.csv'
        if not lat_lon_file.exists():
            lat_lon_df = integrated_df[['lat', 'lon']] #NOTE: duplicated lat/lon values are not removed
            lat_lon_df.to_csv(lat_lon_file, index=False)
            print(f"Latitude and longitude data saved to: {lat_lon_file}")
        else:
            print(f"File already exists: {lat_lon_file}, skipping save.")
        
        # Return the integrated DataFrame
        return integrated_df
    else:
        print("No data files were found.")
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
                       default=['grassland/grassland_component_df.csv', 'forc/forest_component_df.csv'],
                       help='List of data files to aggregate (default: grassland and forest files)')
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
