"""

Reference: https://g.co/gemini/share/7596a84cd06c

"""

import geopandas
import matplotlib.pyplot as plt

# --- Option 1: Using geopandas (Recommended for geospatial operations) ---
def process_geojson_with_geopandas(filepath="bioregions.geojson"):
    """
    Reads, inspects, and performs basic operations on a GeoJSON file using geopandas.
    """
    try:
        # Read the GeoJSON file
        gdf = geopandas.read_file(filepath)

        print("--- Geopandas: Successfully loaded GeoJSON ---")

        # 1. Basic Information
        print(f"\nNumber of features: {len(gdf)}")
        print(f"Coordinate Reference System (CRS): {gdf.crs}")
        print("\nFirst 5 features (head):")
        print(gdf.head())

        # 2. Inspecting Columns (Properties and Geometry)
        print("\nAvailable columns (properties + geometry):")
        print(gdf.columns)

        # 3. Accessing Geometry
        # The 'geometry' column holds the geometric objects (polygons, points, etc.)
        print(f"\nType of geometry column: {type(gdf.geometry)}")
        print(f"First geometry object: {gdf.geometry.iloc[0]}")

        # 4. Accessing Properties (Attributes)
        # Let's assume your GeoJSON has properties like 'name' or 'id'.
        # You'll need to inspect gdf.columns or gdf.head() to see actual property names.
        if not gdf.empty:
            print("\n--- Accessing Properties of the first feature ---")
            first_feature_properties = gdf.iloc[0]
            for col_name in gdf.columns:
                if col_name != 'geometry': # Exclude the geometry object itself from this print
                    print(f"  Property '{col_name}': {first_feature_properties[col_name]}")

            # Example: Access a specific property for all features (if 'name' column exists)
            # Replace 'name' with an actual property name from your file
            if 'name' in gdf.columns:
                print("\n--- Example: Listing all 'name' properties (first 10) ---")
                print(gdf['name'].head(10))
            elif 'NAME' in gdf.columns: # Or try uppercase
                print("\n--- Example: Listing all 'NAME' properties (first 10) ---")
                print(gdf['NAME'].head(10))
            else:
                print("\nNote: A 'name' or 'NAME' property was not found. Please inspect your file's columns.")


        # 5. Simple Filtering (Example)
        # This is a generic example. You'll need to adapt 'property_name' and 'value_to_filter_by'
        # For example, if you have a property 'eco_id' and want to find features where it's '123':
        # filtered_gdf = gdf[gdf['eco_id'] == '123']
        #
        # Let's try to filter based on a common property if it exists, like 'BIOME_NAME' from some bioregion files.
        # Please adapt this to your file's actual properties.
        property_to_filter = None
        if 'BIOME_NAME' in gdf.columns:
            property_to_filter = 'BIOME_NAME'
            # Get the first biome name to use for filtering example
            if not gdf[property_to_filter].empty:
                value_to_filter_by = gdf[property_to_filter].iloc[0]
                filtered_gdf = gdf[gdf[property_to_filter] == value_to_filter_by]
                print(f"\n--- Example: Filtering features where '{property_to_filter}' is '{value_to_filter_by}' ---")
                print(f"Number of features after filtering: {len(filtered_gdf)}")
                print(filtered_gdf.head())
            else:
                print(f"\nColumn '{property_to_filter}' is empty, skipping filter example.")

        elif 'OBJECTID' in gdf.columns: # A generic ID column
            property_to_filter = 'OBJECTID'
            if not gdf[property_to_filter].empty:
                # Example: filter features where OBJECTID is less than a certain value
                # This assumes OBJECTID is numeric. Convert if necessary.
                try:
                    gdf[property_to_filter] = gdf[property_to_filter].astype(int)
                    value_to_filter_by = gdf[property_to_filter].min() + 1 # Example value
                    if gdf[property_to_filter].min() < value_to_filter_by: # ensure filter makes sense
                        filtered_gdf = gdf[gdf[property_to_filter] < value_to_filter_by]
                        print(f"\n--- Example: Filtering features where '{property_to_filter}' < {value_to_filter_by} ---")
                        print(f"Number of features after filtering: {len(filtered_gdf)}")
                        print(filtered_gdf.head())
                    else:
                        print(f"\nCould not apply meaningful filter on '{property_to_filter}'.")
                except ValueError:
                    print(f"\nCould not convert '{property_to_filter}' to numeric for filtering example.")
            else:
                print(f"\nColumn '{property_to_filter}' is empty, skipping filter example.")
        else:
            print("\nSkipping filtering example as common properties like 'BIOME_NAME' or 'OBJECTID' were not found. "
                  "Inspect your file's columns and adapt the filtering logic.")


        # 6. Iterating through features
        print("\n--- Iterating through the first 3 features (geopandas) ---")
        for index, row in gdf.head(3).iterrows():
            print(f"\nFeature index: {index}")
            # Access properties by column name
            # Again, replace 'name' with actual property names from your file
            if 'name' in row:
                print(f"  Name: {row['name']}")
            elif 'NAME' in row:
                 print(f"  Name: {row['NAME']}")

            # Access geometry
            # print(f"  Geometry type: {row.geometry.geom_type}")
            # print(f"  Geometry coordinates (excerpt): {row.geometry.exterior.coords[:5] if row.geometry.geom_type == 'Polygon' else 'N/A for this type'}")
            print(f"  Geometry: {row.geometry.geom_type} (Area: {row.geometry.area if hasattr(row.geometry, 'area') else 'N/A'})")


    except ImportError:
        print("Geopandas library not found. Please install it: pip install geopandas")
    except Exception as e:
        print(f"An error occurred with geopandas: {e}")
        print("Ensure the filepath is correct and the file is a valid GeoJSON.")


def plot_static_map(gdf):
    if gdf is None or gdf.empty:
        print("GeoDataFrame is empty or not loaded. Cannot plot static map.")
        return

    # --- Basic Plot ---
    # This will plot all geometries with a default style
    print("\nDisplaying basic static plot...")
    gdf.plot(figsize=(10, 7))
    plt.title("Basic Plot of Bioregions")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()

    # --- Plot with Colors based on a Categorical Property ---
    # Replace 'BIOME_NAME' with a categorical column from your data.
    categorical_column = 'Bioregions' # EXAMPLE - CHANGE IF NEEDED
    if categorical_column in gdf.columns:
        print(f"\nDisplaying static plot colored by '{categorical_column}'...")
        gdf.plot(column=categorical_column, legend=True, figsize=(12, 8), legend_kwds={'title': categorical_column, 'loc': 'lower left', 'bbox_to_anchor':(1,0)})
        plt.title(f"Bioregions Colored by {categorical_column}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.tight_layout() # Adjust layout to make room for legend
        plt.show()
    else:
        print(f"\nColumn '{categorical_column}' not found. Skipping categorical plot.")
        print(f"Available columns for plotting: {gdf.columns.tolist()}")


    # --- Choropleth Map based on a Numerical Property ---
    # First, ensure you have a numerical column. If not, you can calculate one, e.g., area.
    # Geopandas can calculate area. The units will depend on the CRS.
    # For geographic CRS (like EPSG:4326), area is in square degrees, which is not ideal for visualization.
    # It's better to use an equal-area projection first if you need accurate areas for analysis,
    # but for a simple plot, we can use it directly or use an existing numerical column.

    numerical_column = 'AREA_KM2' # EXAMPLE - CHANGE IF NEEDED, or use calculated area
    if numerical_column not in gdf.columns and 'geometry' in gdf.columns:
        print(f"'{numerical_column}' not found. Attempting to calculate area (may not be in km^2)...")
        # Make sure the CRS is appropriate for area calculation if you need meaningful units.
        # For visualization purposes here, raw area from the current CRS might be okay.
        # To get area in square meters, you might first project to an equal area CRS:
        # gdf_projected = gdf.to_crs("EPSG:3395") # Example: World Mercator (not equal area but commonly used)
        # gdf['calculated_area'] = gdf_projected.area / 10**6 # Area in km^2 (approx if Mercator)
        # For simplicity, we'll use the area from the existing CRS if no area column exists
        try:
            gdf['calculated_area'] = gdf.area
            numerical_column = 'calculated_area'
            print("Using 'calculated_area'. Note: Units depend on the CRS.")
        except Exception as e:
            print(f"Could not calculate area: {e}")
            numerical_column = None


    if numerical_column and numerical_column in gdf.columns:
        print(f"\nDisplaying choropleth map based on '{numerical_column}'...")
        gdf.plot(column=numerical_column, cmap='viridis', legend=True, figsize=(12, 8),
                 legend_kwds={'label': f"Value of {numerical_column}", 'orientation': "horizontal"})
        plt.title(f"Bioregions Choropleth Map ({numerical_column})")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.show()
    else:
        print(f"\nNumerical column '{numerical_column}' not found or calculable. Skipping choropleth plot.")
        print(f"Available columns for plotting: {gdf.columns.tolist()}")

    # --- Adding a Basemap (Optional, requires contextily) ---
    # For this, you might need to install contextily: pip install contextily
    # Also, ensure your data is in Web Mercator projection (EPSG:3857)
    try:
        import contextily as cx
        print("\nDisplaying plot with a basemap (requires contextily)...")
        # Ensure data is in Web Mercator for most contextily providers
        gdf_web_mercator = gdf.to_crs(epsg=3857)
        ax = gdf_web_mercator.plot(figsize=(12, 8), alpha=0.5, edgecolor='k') # k is black
        cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik) # Or other providers
        ax.set_title("Bioregions with OpenStreetMap Basemap")
        ax.set_axis_off() # Hide axis for basemaps
        plt.show()
    except ImportError:
        print("\n`contextily` library not found. Skipping basemap example. Install with `pip install contextily`.")
    except Exception as e:
        print(f"\nCould not generate plot with basemap: {e}")


####
if __name__ == "__main__":
    # IMPORTANT: Replace "bioregions.geojson" with the actual path to your downloaded file
    # For example, if it's in a 'data' subdirectory: geojson_file_path = "data/bioregions.geojson"
    # Or if it has a different name: geojson_file_path = "my_downloaded_bioregions.geojson"

    # Make sure this file exists in the same directory as the script, or provide the full path.
    geojson_file_path = "../landcover/one_earth-bioregions-2023.geojson" 

    print(f"Attempting to process GeoJSON file: {geojson_file_path}")

    # Using geopandas (recommended)
    process_geojson_with_geopandas(geojson_file_path)

    print("\n" + "="*50 + "\n")

    print("\n--- Script Finished ---")
    print("Remember to inspect your GeoJSON file's structure (e.g., property names) to adapt the script as needed.")

    try:
        gdf = geopandas.read_file(geojson_file_path)
        print("GeoJSON file loaded successfully into a GeoDataFrame.")
        print(f"Number of features: {len(gdf)}")
        print(f"Available columns: {gdf.columns.tolist()}")
        # It's crucial to know your CRS, especially for combining with other data or basemaps
        print(f"Coordinate Reference System (CRS): {gdf.crs}")
        # If your data is not in WGS84 (EPSG:4326), you might want to convert it for web mapping:
        # if gdf.crs != "EPSG:4326":
        #     print("Converting CRS to EPSG:4326 for web mapping compatibility...")
        #     gdf = gdf.to_crs("EPSG:4326")
        #     print(f"New CRS: {gdf.crs}")

    except FileNotFoundError:
        print(f"Error: The file '{geojson_file_path}' was not found.")
        gdf = None
    except Exception as e:
        print(f"An error occurred while loading the GeoJSON: {e}")
        gdf = None

    # --- Run the static plotting function if gdf is loaded ---
    if gdf is not None:
        plot_static_map(gdf)