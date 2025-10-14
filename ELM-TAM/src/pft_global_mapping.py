#!/usr/bin/env python3
"""
Script to read ELM surface data file and create global maps of Plant Functional Types (PFTs)
Based on surfdata_360x720cru_simyr1850_c180216.nc format
"""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import netCDF4
from netCDF4 import Dataset
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
from pathlib import Path
import matplotlib.patches as mpatches

class PFTMapper:
    """Class for reading and mapping Plant Functional Types from ELM surface data"""
    
    def __init__(self, surfdata_path, sites_csv_path=None):
        """Initialize with path to surface data file and optional TAM sites CSV"""
        self.surfdata_path = surfdata_path
        self.sites_csv_path = sites_csv_path
        self.surfdata = None
        self.sites_data = None
        self.pft_names = {
            0: "bare soil",
            1: "needleleaf evergreen temperate tree",
            2: "needleleaf evergreen boreal tree", 
            3: "needleleaf deciduous boreal tree",
            4: "broadleaf evergreen tropical tree",
            5: "broadleaf evergreen temperate tree",
            6: "broadleaf deciduous tropical tree",
            7: "broadleaf deciduous temperate tree",
            8: "broadleaf deciduous boreal tree",
            9: "broadleaf evergreen shrub",
            10: "broadleaf deciduous temperate shrub",
            11: "broadleaf deciduous boreal shrub",
            12: "C3 arctic grass",
            13: "C3 non-arctic grass",
            14: "C4 grass",
            15: "C3 crop",
            16: "C4 crop"
        }
        
    def load_data(self):
        """Load surface data from NetCDF file"""
        if not os.path.exists(self.surfdata_path):
            raise FileNotFoundError(f"Surface data file not found: {self.surfdata_path}")
        
        print(f"Loading surface data from: {self.surfdata_path}")
        self.surfdata = Dataset(self.surfdata_path, 'r')
        
        print(f"Grid dimensions: {self.surfdata.dimensions['lsmlon'].size} x {self.surfdata.dimensions['lsmlat'].size}")
        print(f"Number of PFTs: {self.surfdata.dimensions['natpft'].size}")
        
    def load_sites_data(self):
        """Load TAM sites data from CSV file"""
        if self.sites_csv_path and os.path.exists(self.sites_csv_path):
            print(f"Loading TAM sites data from: {self.sites_csv_path}")
            self.sites_data = pd.read_csv(self.sites_csv_path)
            print(f"Loaded {len(self.sites_data)} TAM sites")
        else:
            print("No TAM sites data provided or file not found")
        
    def get_pft_data(self, pft_index):
        """Get PFT percentage data for a specific PFT index"""
        if self.surfdata is None:
            self.load_data()
            
        return self.surfdata.variables['PCT_NAT_PFT'][pft_index, :, :]
    
    def get_coordinates(self):
        """Get longitude and latitude coordinates"""
        if self.surfdata is None:
            self.load_data()
            
        return self.surfdata.variables['LONGXY'][:], self.surfdata.variables['LATIXY'][:]
    
    def add_tam_sites_overlay(self, ax, marker_size=50, legend=True, filter_pft=None):
        """Add TAM sites as colored markers based on their PFT and mycorrhizal type"""
        if self.sites_data is None:
            self.load_sites_data()

        if self.sites_data is None:
            return

        # Define colors for different mycorrhizal types
        fungi_colors = {
            'EMF': 'red',
            'EM': 'red',
            'AM': 'blue',
            'EM/AM': 'purple'
        }

        # Define markers for different PFT categories
        pft_markers = {
            1: '^',  # needleleaf evergreen temperate tree
            2: '^',  # needleleaf evergreen boreal tree
            3: '^',  # needleleaf deciduous boreal tree
            4: 's',  # broadleaf evergreen tropical tree
            5: 's',  # broadleaf evergreen temperate tree
            6: 'o',  # broadleaf deciduous tropical tree
            7: 'o',  # broadleaf deciduous temperate tree
            8: 'o',  # broadleaf deciduous boreal tree
            9: 'd',  # broadleaf evergreen shrub
            10: 'd', # broadleaf deciduous temperate shrub
            11: 'd', # broadleaf deciduous boreal shrub
            12: 'v', # C3 arctic grass
            13: 'v', # C3 non-arctic grass
            14: 'v', # C4 grass
        }

        # Plot sites grouped by mycorrhizal type and PFT
        plotted_combinations = set()

        for _, site in self.sites_data.iterrows():
            if pd.isna(site['Lat']) or pd.isna(site['Lon']) or pd.isna(site['PFTs']):
                continue

            fungi_type = site['Fungi']
            pft_type = int(site['PFTs'])

            # Filter sites by PFT if specified
            if filter_pft is not None and pft_type != filter_pft:
                continue

            color = fungi_colors.get(fungi_type, 'gray')
            marker = pft_markers.get(pft_type, 'o')

            # Plot the site
            ax.scatter(site['Lon'], site['Lat'],
                      c=color, marker=marker, s=marker_size,
                      edgecolors='black', linewidth=0.5,
                      transform=ccrs.PlateCarree(),
                      zorder=10)

            # Track what we've plotted for legend
            plotted_combinations.add((fungi_type, color, pft_type, marker))

        # Add legend if requested
        if legend:
            self._add_sites_legend(ax, plotted_combinations)
    
    def _add_sites_legend(self, ax, plotted_combinations):
        """Add legend for TAM sites overlay"""
        # Create legend elements
        legend_elements = []
        
        # Group by mycorrhizal type
        fungi_groups = {}
        for fungi_type, color, pft_type, marker in plotted_combinations:
            if fungi_type not in fungi_groups:
                fungi_groups[fungi_type] = []
            fungi_groups[fungi_type].append((pft_type, marker))
        
        # Create legend entries
        for fungi_type, sites_info in fungi_groups.items():
            # Get unique markers for this fungi type
            unique_markers = list(set([marker for _, marker in sites_info]))
            
            if len(unique_markers) == 1:
                # Single marker type for this fungi group
                marker = unique_markers[0]
                color = {'EMF': 'red', 'EM': 'red', 'AM': 'blue', 'EM/AM': 'purple'}.get(fungi_type, 'gray')
                legend_elements.append(
                    plt.Line2D([0], [0], marker=marker, color='w', 
                              markerfacecolor=color, markersize=8,
                              markeredgecolor='black', markeredgewidth=0.5,
                              label=f'{fungi_type}')
                )
            else:
                # Multiple marker types - show them all
                for marker in unique_markers:
                    pft_names_short = {'^': 'Trees-NL', 's': 'Trees-BL-Ever', 
                                     'o': 'Trees-BL-Dec', 'd': 'Shrubs', 'v': 'Grasses'}
                    marker_name = pft_names_short.get(marker, 'Other')
                    color = {'EMF': 'red', 'EM': 'red', 'AM': 'blue', 'EM/AM': 'purple'}.get(fungi_type, 'gray')
                    legend_elements.append(
                        plt.Line2D([0], [0], marker=marker, color='w',
                                  markerfacecolor=color, markersize=8,
                                  markeredgecolor='black', markeredgewidth=0.5,
                                  label=f'{fungi_type}-{marker_name}')
                    )
        
        # Add legend to plot
        ax.legend(handles=legend_elements, loc='upper left',
                 bbox_to_anchor=(0.02, 0.98), fontsize=8,
                 title='TAM Sites', title_fontsize=9)

    def _add_site_labels(self, ax, filter_pft=None):
        """Add FluxNet site ID labels for TAM sites"""
        if self.sites_data is None:
            return

        for _, site in self.sites_data.iterrows():
            if pd.isna(site['Lat']) or pd.isna(site['Lon']) or pd.isna(site['PFTs']):
                continue

            pft_type = int(site['PFTs'])

            # Filter sites by PFT if specified
            if filter_pft is not None and pft_type != filter_pft:
                continue

            # Add site label with FluxNet ID
            ax.annotate(site['Site ID'],
                       (site['Lon'], site['Lat']),
                       xytext=(8, 8), textcoords='offset points',
                       fontsize=9, fontweight='bold',
                       color='black',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                alpha=0.8, edgecolor='black', linewidth=0.5),
                       transform=ccrs.PlateCarree(),
                       zorder=15)

    def plot_single_pft(self, pft_index, save_path=None, figsize=(12, 8), dpi=200, overlay_sites=True):
        """Create a global map for a single PFT"""
        if pft_index not in self.pft_names:
            raise ValueError(f"Invalid PFT index: {pft_index}")
        
        pft_data = self.get_pft_data(pft_index)
        lon, lat = self.get_coordinates()
        
        plt.figure(figsize=figsize, dpi=dpi, facecolor='w', edgecolor='k')
        
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_global()
        ax.coastlines(resolution='50m', linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.3)
        
        # Add gridlines
        gl = ax.gridlines(linestyle='--', alpha=0.5)
        gl.bottom_labels = True
        gl.left_labels = True
        gl.ylabel_style = {'size': 10}
        gl.xlabel_style = {'size': 10}
        
        # Plot PFT data
        im = plt.pcolormesh(lon, lat, pft_data, 
                           cmap='YlOrRd', 
                           transform=ccrs.PlateCarree(),
                           vmin=0, vmax=np.nanmax(pft_data))
        
        # Add TAM sites overlay (only sites matching this PFT)
        if overlay_sites:
            self.add_tam_sites_overlay(ax, marker_size=60, filter_pft=pft_index, legend=False)
            # Add site labels for corresponding PFT sites
            self._add_site_labels(ax, filter_pft=pft_index)
        
        # Add colorbar
        cbar = plt.colorbar(im, orientation='horizontal',
                           pad=0.08, shrink=0.8, aspect=30)
        cbar.set_label('Percentage (%)', fontsize=12)
        
        plt.title(f"PFT {pft_index}: {self.pft_names[pft_index]}", 
                 fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved PFT {pft_index} map to: {save_path}")
        
        plt.show()
    
    def plot_dominant_pft(self, save_path=None, figsize=(15, 10), dpi=200, overlay_sites=True):
        """Create a map showing the dominant PFT at each grid cell"""
        if self.surfdata is None:
            self.load_data()
        
        # Get all PFT data
        all_pft_data = self.surfdata.variables['PCT_NAT_PFT'][:]
        lon, lat = self.get_coordinates()
        
        # Find dominant PFT at each grid cell
        dominant_pft = np.argmax(all_pft_data, axis=0)
        max_pft_percent = np.max(all_pft_data, axis=0)
        
        # Mask areas with very low vegetation coverage
        mask = max_pft_percent < 5  # Less than 5% vegetation
        dominant_pft_masked = np.ma.masked_where(mask, dominant_pft)
        
        plt.figure(figsize=figsize, dpi=dpi, facecolor='w', edgecolor='k')
        
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_global()
        ax.coastlines(resolution='50m', linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.3)
        
        # Use a discrete colormap
        cmap = plt.colormaps.get_cmap('tab20').resampled(17)  # 17 PFTs
        
        im = plt.pcolormesh(lon, lat, dominant_pft_masked,
                           cmap=cmap, vmin=0, vmax=16,
                           transform=ccrs.PlateCarree())
        
        # Add TAM sites overlay
        if overlay_sites:
            self.add_tam_sites_overlay(ax, marker_size=80)
        
        # Create custom colorbar with PFT names
        cbar = plt.colorbar(im, orientation='vertical', 
                           pad=0.02, shrink=0.8, aspect=30)
        
        # Set colorbar ticks and labels
        cbar.set_ticks(range(17))
        pft_labels = [f"{i}: {self.pft_names[i][:20]}" for i in range(17)]
        cbar.set_ticklabels(pft_labels, fontsize=8)
        cbar.set_label('Dominant Plant Functional Type', fontsize=12)
        
        plt.title("Global Distribution of Dominant Plant Functional Types", 
                 fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved dominant PFT map to: {save_path}")
        
        plt.show()
    
    def plot_vegetation_summary(self, save_path=None, figsize=(20, 12), dpi=200):
        """Create a summary plot showing multiple PFT categories"""
        if self.surfdata is None:
            self.load_data()
        
        all_pft_data = self.surfdata.variables['PCT_NAT_PFT'][:]
        lon, lat = self.get_coordinates()
        
        # Define PFT categories
        categories = {
            'Trees (Total)': [1, 2, 3, 4, 5, 6, 7, 8],
            'Needleleaf Trees': [1, 2, 3],
            'Broadleaf Trees': [4, 5, 6, 7, 8],
            'Shrubs': [9, 10, 11],
            'Grasses': [12, 13, 14],
            'Crops': [15, 16]
        }
        
        fig, axes = plt.subplots(2, 3, figsize=figsize, 
                                subplot_kw={'projection': ccrs.PlateCarree()})
        axes = axes.flatten()
        
        for idx, (category, pft_indices) in enumerate(categories.items()):
            ax = axes[idx]
            
            # Sum PFT percentages for this category
            category_data = np.sum(all_pft_data[pft_indices, :, :], axis=0)
            
            ax.set_global()
            ax.coastlines(resolution='50m', linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.3)
            
            im = ax.pcolormesh(lon, lat, category_data,
                              cmap='YlGn', vmin=0, vmax=100,
                              transform=ccrs.PlateCarree())
            
            ax.set_title(category, fontsize=12, fontweight='bold')
            
            # Add colorbar for each subplot
            cbar = plt.colorbar(im, ax=ax, orientation='horizontal',
                               pad=0.05, shrink=0.8, aspect=20)
            cbar.set_label('Percentage (%)', fontsize=10)
        
        plt.suptitle('Global Distribution of Plant Functional Type Categories', 
                     fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved vegetation summary to: {save_path}")
        
        plt.show()
    
    def export_pft_statistics(self, output_file='pft_global_statistics.csv'):
        """Export global statistics for each PFT to CSV"""
        if self.surfdata is None:
            self.load_data()
        
        all_pft_data = self.surfdata.variables['PCT_NAT_PFT'][:]
        area_data = self.surfdata.variables['AREA'][:]
        
        stats = []
        for pft_idx in range(17):
            pft_percent = all_pft_data[pft_idx, :, :]
            
            # Calculate statistics
            total_area = np.sum(pft_percent * area_data / 100)  # km²
            mean_percent = np.nanmean(pft_percent)
            max_percent = np.nanmax(pft_percent)
            std_percent = np.nanstd(pft_percent)
            
            # Count grid cells with >5% coverage
            significant_cells = np.sum(pft_percent > 5)
            
            stats.append({
                'PFT_Index': pft_idx,
                'PFT_Name': self.pft_names[pft_idx],
                'Total_Area_km2': total_area,
                'Mean_Percent': mean_percent,
                'Max_Percent': max_percent,
                'Std_Percent': std_percent,
                'Significant_Cells': significant_cells
            })
        
        df = pd.DataFrame(stats)
        df.to_csv(output_file, index=False)
        print(f"PFT statistics exported to: {output_file}")
        return df
    
    def plot_tam_sites_distribution(self, save_path=None, figsize=(15, 10), dpi=200):
        """Create a dedicated map showing TAM sites distribution with detailed legend"""
        if self.sites_data is None:
            self.load_sites_data()
        
        if self.sites_data is None:
            print("No TAM sites data available for plotting")
            return
        
        plt.figure(figsize=figsize, dpi=dpi, facecolor='w', edgecolor='k')
        
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_global()
        ax.coastlines(resolution='50m', linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
        ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
        
        # Add gridlines
        gl = ax.gridlines(linestyle='--', alpha=0.5)
        gl.bottom_labels = True
        gl.left_labels = True
        gl.ylabel_style = {'size': 10}
        gl.xlabel_style = {'size': 10}
        
        # Add TAM sites with larger markers and detailed legend
        self.add_tam_sites_overlay(ax, marker_size=100, legend=True)
        
        # Add site labels for key sites
        key_sites = ['US-Ho1', 'US-MOz']  # Main calibration sites
        for _, site in self.sites_data.iterrows():
            if site['Site ID'] in key_sites:
                ax.annotate(site['Site ID'], 
                           (site['Lon'], site['Lat']), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=10, fontweight='bold',
                           transform=ccrs.PlateCarree())
        
        plt.title("TAM FluxNet Sites Global Distribution\n" +
                 f"Total: {len(self.sites_data)} sites across major biomes", 
                 fontsize=16, fontweight='bold')
        
        # Add text box with site statistics
        site_stats = f"""
        Site Statistics:
        • Total Sites: {len(self.sites_data)}
        • EMF Sites: {len(self.sites_data[self.sites_data['Fungi'] == 'EMF'])}
        • EM Sites: {len(self.sites_data[self.sites_data['Fungi'] == 'EM'])}
        • AM Sites: {len(self.sites_data[self.sites_data['Fungi'] == 'AM'])}
        • EM/AM Sites: {len(self.sites_data[self.sites_data['Fungi'] == 'EM/AM'])}
        """
        
        ax.text(0.02, 0.02, site_stats.strip(), transform=ax.transAxes,
                fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved TAM sites distribution map to: {save_path}")
        
        plt.show()

def main():
    """Main function to demonstrate PFT mapping functionality"""
    
    # Define paths to data files
    surfdata_file = "../data/parameter/tam_params/surfdata_360x720cru_simyr1850_c180216.nc"
    sites_file = "../data/parameter/sites/tam_fluxnet_sites.csv"
    
    # Check if files exist
    if not os.path.exists(surfdata_file):
        print(f"Surface data file not found: {surfdata_file}")
        print("Please check the file path or download the required data.")
        return
    
    # Initialize mapper with both surface data and sites
    mapper = PFTMapper(surfdata_file, sites_file)
    
    try:
        # Load data
        mapper.load_data()
        
        # Create output directory
        output_dir = Path("../data/pft_maps")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot TAM sites distribution
        print("Creating TAM sites distribution map...")
        mapper.plot_tam_sites_distribution(save_path=output_dir / "tam_sites_distribution.png")
        
        # Plot dominant PFT map with TAM sites overlay
        print("Creating dominant PFT map with TAM sites overlay...")
        mapper.plot_dominant_pft(save_path=output_dir / "dominant_pft_global_with_sites.png")
        
        # Plot vegetation category summary
        print("Creating vegetation category summary...")
        mapper.plot_vegetation_summary(save_path=output_dir / "vegetation_categories.png")
        
        # Plot a few individual PFTs as examples
        #example_pfts = [1, 2, 4, 7, 13, 14]  # Boreal evergreen, tropical broadleaf, C3 grass, C4 grass
        all_pfts = list(mapper.pft_names.keys())
        for pft_idx in all_pfts:
            print(f"Creating map for PFT {pft_idx}: {mapper.pft_names[pft_idx]} with corresponding TAM sites")
            safe_name = mapper.pft_names[pft_idx].replace(" ", "_").replace("/", "_")
            mapper.plot_single_pft(pft_idx,
                                 save_path=output_dir / f"pft_{pft_idx:02d}_{safe_name}_with_corresponding_sites.png")
        
        # Export statistics
        print("Exporting PFT statistics...")
        stats_df = mapper.export_pft_statistics(output_dir / "pft_global_statistics.csv")
        print("\nGlobal PFT Statistics Summary:")
        print(stats_df.to_string(index=False))
        
        print(f"\nAll outputs saved to: {output_dir.absolute()}")
        
    except Exception as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    main()