""" Integrate data from different sources into a single dataset for machine learing.

    This script fetches data from various sources, processes it, and integrates it into a single DataFrame.
    The data sources include:
    - Dryad
    - ForC
    - ELM-TAM
"""

import pandas as pd
import os
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def integrate_data():
    """Integrate data from different sources into a single DataFrame."""
    
    # Define and validate the data directory path
    data_dir = Path('../productivity').resolve()

    # Verify the directory exists
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    print(f"Using data directory: {data_dir}")
    
    # Initialize an empty list to hold DataFrames
    dataframes = []
    
    # List of data files to read
    data_files = [
        'grassland/grassland_component_df.csv',
        'forc/forest_component_df.csv'
    ]
    
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
        return integrated_df
    else:
        print("No data files were found.")
        return pd.DataFrame()  # Return an empty DataFrame if no files were found


def plot_data_distribution(df):
    """ Plot the spatial distribution of data points

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
    output_path = os.path.join(output_dir, 'integrated_data_distribution.pdf')
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


if __name__ == "__main__":
    try:
        integrated_data = integrate_data()
        if not integrated_data.empty:
            print(f"Integrated DataFrame shape: {integrated_data.shape}")
            print(integrated_data.head())
        else:
            print("No data to display.")
    except Exception as e:
        print(f"An error occurred during data integration: {str(e)}")

    # Plot the integrated data if needed
    plot_data_distribution(integrated_data)
