""" 
The Global biomes boundaries were delineated using the World Wildlife Fund terrestrial ecoregions map. 
For the purpose of the sampling analysis, the biome was partitioned into a sampling frame consisting of 
square blocks 18.5 km per side. The shapefile contains the sampling frame boundaries.

Data: https://glad.umd.edu/dataset/gfm/globaldata/global-data

Reference: https://g.co/gemini/share/4a73b82c0b75

"""
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# Replace 'your_shapefile.shp' with the actual path to your shapefile
shapefile_path = '../landcover/data/biomes/biomes.shp'

try:
    # Read the shapefile
    gdf = gpd.read_file(shapefile_path)

    # Print the first few rows of the attribute table
    print("First 5 rows of the attribute table:")
    print(gdf.head())

    # Print the number of features (rows) and attributes (columns)
    print(f"\nNumber of features: {len(gdf)}")
    print(f"Number of attributes: {len(gdf.columns)}")

    # Print the coordinate reference system (CRS)
    print(f"\nCoordinate Reference System (CRS): {gdf.crs}")

    # Access geometry of the first feature
    # print("\nGeometry of the first feature:")
    # print(gdf.geometry.iloc[0])

    # You can also plot the shapefile (requires matplotlib)
    # gdf.plot()
    # plt.title("My Shapefile")
    # plt.xlabel("Longitude")
    # plt.ylabel("Latitude")
    # plt.show()

    # Create figure and axis with specified size
    fig, ax = plt.subplots(figsize=(15, 10),
                           subplot_kw={'projection': ccrs.PlateCarree()})

    # Add natural earth features
    ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=0.5)
    #ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray')
    #ax.add_feature(cfeature.STATES, linestyle=':', edgecolor='gray', alpha=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.5)
    ax.add_feature(cfeature.LAKES, facecolor='lightblue', alpha=0.5)

    # Plot the geodataframe with customized appearance
    gdf.plot(ax=ax,
             column='BIOME',           # Color by BIOME field
             legend=True,              # Show legend
             legend_kwds={'title': 'Biome',
                         'bbox_to_anchor': (0.05, .25),
                         'loc': 'lower left'},
             cmap='Spectral',         # Color scheme
             alpha=0.7,               # Transparency
             edgecolor='black',       # Border color
             linewidth=0.5,           # Border width
             transform=ccrs.PlateCarree()
             )           

    # Customize the plot
    plt.title("Global Biomes Distribution", 
             fontsize=16, 
             pad=20)
    plt.xlabel("Longitude", fontsize=12)
    plt.ylabel("Latitude",  fontsize=12)

    # Add gridlines
    ax.grid(True, linestyle='--', alpha=0.6)

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()

    # Create output directory if it doesn't exist
    output_dir = '../landcover/figures'
    os.makedirs(output_dir, exist_ok=True)

    # Save figure as PDF with high resolution
    output_path = os.path.join(output_dir, 'global_biomes_map.pdf')
    plt.savefig(output_path, 
                format='pdf',
                dpi=300,              # High resolution
                bbox_inches='tight',  # Prevent cutoff
                facecolor='white',    # White background
                edgecolor='none',     # No edge color
                pad_inches=0.1)       # Add small padding

    print(f"\nMap saved as: {output_path}")

    # Show the plot
    plt.show()

except FileNotFoundError:
    print(f"Error: The file '{shapefile_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")