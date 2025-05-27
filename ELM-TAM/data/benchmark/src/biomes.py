""" 
Data: https://glad.umd.edu/dataset/gfm/globaldata/global-data

Reference: https://g.co/gemini/share/4a73b82c0b75

"""


import geopandas as gpd
import matplotlib.pyplot as plt

# Replace 'your_shapefile.shp' with the actual path to your shapefile
shapefile_path = '../landcover/biomes/biomes.shp'

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
    fig, ax = plt.subplots(figsize=(15, 10))

    # Plot the geodataframe with customized appearance
    gdf.plot(ax=ax,
             column='BIOME',           # Color by BIOME field
             legend=True,              # Show legend
             legend_kwds={'title': 'Biome Types',
                         'bbox_to_anchor': (1.05, 1),
                         'loc': 'upper left'},
             cmap='Spectral',         # Color scheme
             alpha=0.7,               # Transparency
             edgecolor='black',       # Border color
             linewidth=0.5)           # Border width

    # Customize the plot
    plt.title("Global Biomes Distribution", 
             fontsize=16, 
             pad=20)
    plt.xlabel("Longitude", fontsize=12)
    plt.ylabel("Latitude", fontsize=12)

    # Add gridlines
    ax.grid(True, linestyle='--', alpha=0.6)

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()

    # Show the plot
    plt.show()

except FileNotFoundError:
    print(f"Error: The file '{shapefile_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")