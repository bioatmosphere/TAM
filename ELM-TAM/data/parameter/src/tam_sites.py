"""ELM-TAM Sensitivity Analysis

This script:

1. downloads a specific tab from a Google Sheet as a CSV file, checks if the file exists, and if not, downloads it.
2. reads the CSV file into a Pandas DataFrame, displays the first few rows, and provides basic information about the dataset.
3. visualizes the FLUXNET sites on a global map using Cartopy, with points colored by their Plant Functional Types (PFTs).
4. overlays the sites on distributions of PFTs, adds text labels for each site, and saves the map as a high-resolution PDF file.

"""

import requests
import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import netCDF4
from netCDF4 import Dataset

# Define a function of download the Google Sheet tab as a CSV file
def download_csv():

    # Download the Google Sheet tab as a CSV file
    # reference: https://www.perplexity.ai/search/how-to-download-a-tab-in-googl-ENw0RfhqQx6l6UnpZ7n7lw
    spreadsheet_id = '197DxMUAzWOde2NbQ4DADmo-aTRpBjVqDbCSyel7whDs'
    gid = '742842737' 
    url = f'https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=csv&gid={gid}'
    
    response = requests.get(url)
    if response.status_code == 200:
        with open('tam_fluxnet_sites.csv', 'wb') as f:
            f.write(response.content)
        print('Tab downloaded as CSV.')
    else:
        print(f'Failed to download: {response.status_code}')


# Check if the CSV file exists
os.chdir('../data/')
csv_file = 'tam_fluxnet_sites.csv'
if os.path.isfile(csv_file):
    print(f"The file '{csv_file}' exists in the directory")
    user_input = input("\nWould you like to update the file? (y/n): ").lower()
    if user_input == 'y':
        print("Downloading updated version...")
        download_csv()
    else:
        print("Using existing file.")
else:
    print(f"The file '{csv_file}' does not exist in the directory")
    download_csv()
    

# Read the CSV file and print the first few rows
sites_df = pd.read_csv('tam_fluxnet_sites.csv')

# Display the first 5 rows of the data
print("\nFirst 5 rows of the CSV file:")
print(sites_df.head())

# Display basic information about the dataset
print("\nDataset info:")
print(sites_df.info())

# Overlay the sites on a gloal map
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
                 linestyle='--'
                 )
gl.top_labels = False  # Remove top labels
gl.right_labels = False  # Remove right labels

#----------------------------------------------------------------
#Plot the data points on ELM surface data
#----------------------------------------------------------------
# Load the NetCDF file containing PFT data
surfdata = Dataset('surfdata_360x720cru_simyr1850_c180216.nc','r')
# Specify the indices of layers you want to plot
layers_to_plot = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]  
for i, layer in enumerate(layers_to_plot):
    ax.pcolormesh(
        surfdata.variables['LONGXY'][:],
        surfdata.variables['LATIXY'][:],
        surfdata.variables['PCT_NAT_PFT'][layer, :, :],
        cmap='viridis',
        alpha=0.3,  # Add transparency to distinguish layers
        vmin=0,
        vmax=100,  # Adjust based on your data range
        shading='auto',
        transform=ccrs.PlateCarree(),
        zorder=i
    )

# Create a color map for unique PFTs
unique_pfts = sites_df['PFTs'].unique()
#colors = plt.cm.Set3(np.linspace(0, 1, len(unique_pfts)))
colors = plt.cm.tab20(np.linspace(0, 1, len(unique_pfts)))
# Plot points for each PFT category
for pft, color in zip(unique_pfts, colors):
    mask = sites_df['PFTs'] == pft
    ax.scatter(
        sites_df.loc[mask, 'Lon'], 
        sites_df.loc[mask, 'Lat'],
        color=color,
        s=50,
        label=pft,
        transform=ccrs.PlateCarree()
    )

    # Add text labels for each point
    for idx, row in sites_df[mask].iterrows():
        ax.text(row['Lon'] + 1, row['Lat'] + 1,  # Offset text slightly
                row['Site ID'],                      # Assuming 'Site' is the column name
                fontsize=8,
                alpha=0.7,
                transform=ccrs.PlateCarree(),
                bbox=dict(facecolor='white',      # White background for text
                         alpha=0.5,               # Semi-transparent
                         edgecolor='none',
                         pad=0.5))
        
# Adjust legend position and size
ax.legend(
    title='PFTs',
    bbox_to_anchor=(0.1, 0.1),
    loc='lower left',
    shadow=True,                # Add shadow
    borderpad=1                 # Padding between legend and frame
    )

# Adjust layout to prevent legend cutoff
plt.tight_layout()

# Set title
ax.set_title('Distribution of FLUXNET sites for ELM-TAM sensivity', fontsize=15)

# # Save and display the plot
# os.chdir('../')
# output_dir = 'figures'
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # Save the figure with high resolution
# plt.savefig(f'{output_dir}/fluxnet_sites_map.pdf', 
#             dpi=300,              # High resolution
#             bbox_inches='tight',  # Prevent cutting off elements
#             facecolor='white',    # White background
#             edgecolor='none')     # No edge color

# Display the plot
plt.show()

# Close the figure to free memory
plt.close()