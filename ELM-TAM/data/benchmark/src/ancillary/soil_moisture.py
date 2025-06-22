"""
Download and preprocess soil moisture data from Google Drive folder.

https://drive.google.com/drive/folders/1bm57jo6yUHGJ0P-sfPwA4NM5VCzSLoUr

This script provides functionality to:
1. Download individual files from Google Drive using file IDs
2. Download entire folders using gdown library 
3. Extract compressed files automatically
4. Visualize soil moisture data

Dependencies:
    pip install gdown xarray matplotlib cartopy tqdm
"""

import os
import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

try:
    import gdown
except ImportError:
    gdown = None



def download_drive_folder(folder_url, output_dir):
    """
    Download entire Google Drive folder using gdown library.
    
    Args:
        folder_url (str): Google Drive folder URL
        output_dir (str): Local directory to save files
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import gdown
    except ImportError:
        print("Error: gdown library not found. Install with: pip install gdown")
        return False
    
    try:
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading Google Drive folder to: {output_dir}")
        print(f"Source: {folder_url}")
        
        # Download the folder
        result = gdown.download_folder(
            folder_url,
            output=output_dir,
            quiet=False,
            use_cookies=False
        )
        
        if result:
            print("✓ Folder download completed successfully")
            
            # List downloaded files
            downloaded_files = []
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    file_path = Path(root) / file
                    downloaded_files.append(str(file_path.relative_to(output_dir)))
            
            print(f"Downloaded {len(downloaded_files)} files:")
            for file in sorted(downloaded_files):
                print(f"  - {file}")
            
            return True
        else:
            print("✗ Folder download failed")
            return False
            
    except Exception as e:
        print(f"Error downloading folder: {e}")
        return False


def download_drive_file(file_id, output_path):
    """
    Download a single file from Google Drive using gdown (more reliable than requests).
    
    Args:
        file_id (str): Google Drive file ID
        output_path (str): Local path where to save the file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import gdown
    except ImportError:
        print("Error: gdown library not found. Install with: pip install gdown")
        return False
    
    try:
        # Create output directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Check if file already exists
        if Path(output_path).exists():
            print(f"File already exists: {output_path}")
            return True
            
        print(f"Downloading file to: {output_path}")
        
        # Download using gdown
        file_url = f"https://drive.google.com/uc?id={file_id}"
        result = gdown.download(file_url, output_path, quiet=False)
        
        if result and Path(output_path).exists():
            file_size = Path(output_path).stat().st_size
            print(f"✓ Download completed: {output_path} ({file_size:,} bytes)")
            return True
        else:
            print(f"✗ Download failed: {output_path}")
            return False
            
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False


def unzip_files(zip_dir, extract_dir):
    """
    Unzip all zip files in a directory with progress tracking.
    
    Args:
        zip_dir (str): Directory containing zip files
        extract_dir (str): Directory to extract files to
    """
    # Convert to Path objects
    zip_dir = Path(zip_dir)
    extract_dir = Path(extract_dir)
    
    # Create extract directory if it doesn't exist
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all zip files
    zip_files = list(zip_dir.glob("*.zip"))
    if not zip_files:
        print(f"No zip files found in {zip_dir}")
        return
    
    print(f"Found {len(zip_files)} zip files")
    
    # Extract each zip file
    for zip_file in tqdm(zip_files, desc="Extracting files"):
        try:
            with zipfile.ZipFile(zip_file) as zf:
                # Create subfolder named after zip file
                subfolder = extract_dir / zip_file.stem
                subfolder.mkdir(exist_ok=True)
                
                # Extract all files with progress
                for member in tqdm(zf.namelist(), 
                                 desc=f"Extracting {zip_file.name}",
                                 leave=False):
                    zf.extract(member, subfolder)
                    
            print(f"Extracted {zip_file.name} to {subfolder}")
            
        except zipfile.BadZipFile:
            print(f"Error: {zip_file} is not a valid zip file")
        except Exception as e:
            print(f"Error extracting {zip_file}: {e}")


def visualize_soil_moisture(file_path):
    """
    Load and visualize soil moisture data from a NetCDF file.
    
    Args:
        file_path (str): Path to the NetCDF file
    """
    try:
        ds = xr.open_dataset(file_path)
        print(ds)

        #annual_mean = ds['sm'].mean(dim='time')
        annual_mean = ds['sm'][:,0,:,:].mean(dim='time')  # Adjust indexing as needed

        # Create figure with specific projection
        fig, ax = plt.subplots(figsize=(15, 10), 
                              subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Plot data with enhanced styling
        pcm = annual_mean.plot(ax=ax,
                             cmap='RdYlBu',          # Better colormap for precipitation
                             transform=ccrs.PlateCarree(),
                             cbar_kwargs={'label': 'Soil Moisture',
                                        'orientation': 'horizontal',
                                        'pad': 0.05})
        
        # Add map features
        ax.coastlines(resolution='50m', color='black', linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.gridlines(draw_labels=True)
        
        # Customize title and labels
        file_name = Path(file_path).stem
        plt.title(f'Mean Annual Soil Moisture - {file_name}', fontsize=14, pad=20)
        plt.tight_layout()
        
        # Save the figure
        output_path = Path(file_path).parent.parent / f"{Path(file_path).stem}_soil_moisture_map.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Map saved to: {output_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"Error loading or visualizing data: {e}")

def main():
    """
    Main function with command line interface for downloading and processing soil moisture data.
    """
    parser = argparse.ArgumentParser(description='Download and process soil moisture data from Google Drive')
    parser.add_argument('--mode', choices=['folder', 'individual', 'extract', 'visualize'], 
                       default='folder',
                       help='Download mode: folder (entire folder), individual (specific files), extract (unzip), or visualize')
    parser.add_argument('--folder-url', type=str,
                       default='https://drive.google.com/drive/folders/1bm57jo6yUHGJ0P-sfPwA4NM5VCzSLoUr',
                       help='Google Drive folder URL')
    parser.add_argument('--output-dir', '-o', type=str,
                       default='../../ancillary/soilmoisture',
                       help='Output directory for downloaded files')
    parser.add_argument('--extract-dir', type=str,
                       default='../../ancillary/soilmoisture/extracted',
                       help='Directory to extract compressed files')
    parser.add_argument('--file-path', type=str,
                       help='Path to NetCDF file for visualization')
    
    args = parser.parse_args()
    
    print("=== Soil Moisture Data Processor ===")
    print(f"Mode: {args.mode}")
    print()
    
    if args.mode == 'folder':
        # Download entire Google Drive folder
        success = download_drive_folder(args.folder_url, args.output_dir)
        if success:
            print(f"\n✓ Folder downloaded to: {args.output_dir}")
            print("Next steps:")
            print("1. Run with --mode extract to unzip any compressed files")
            print("2. Run with --mode visualize to create plots")
        else:
            print("✗ Folder download failed")
    
    elif args.mode == 'individual':
        # Download individual files using the predefined list
        print("Downloading individual files...")
        success_count = 0
        
        # File configurations (keep existing dictionary)
        # NOTE: file IDs are from the Google Drive folder that can be found when
        # you right-click the file and select Share and Copy link, e.g., for ec_ors.nc:
        # https://drive.google.com/file/d/1-UZ7FEHbqoAXHq6Wa3b5zme8vPeX02kf/view?usp=drive_link
        files_to_download = {
            "ec_ors.nc":  "1-0Ze2XfDWyOqgxpetQqf5HeVxDxX_y7_",
            "olc_ors.nc": "1-UZ7FEHbqoAXHq6Wa3b5zme8vPeX02kf"
        }

        for filename, file_id in files_to_download.items():
            output_path = Path(args.output_dir) / filename
            if download_drive_file(file_id, str(output_path)):
                success_count += 1
        
        print(f"\n✓ Successfully downloaded {success_count}/{len(files_to_download)} files")
    
    elif args.mode == 'extract':
        # Extract compressed files
        print(f"Extracting files from: {args.output_dir}")
        unzip_files(args.output_dir, args.extract_dir)
        print(f"✓ Files extracted to: {args.extract_dir}")
    
    elif args.mode == 'visualize':
        # Visualize soil moisture data
        if args.file_path:
            file_path = args.file_path
        else:
            # Process all netCDF files
            nc_files = list(Path(args.extract_dir).glob("*.nc"))
            file_path = nc_files if nc_files else [Path(args.extract_dir) / "olc_ors.nc"]
        
        if isinstance(file_path, list):
            for nc_file in file_path:
                if Path(nc_file).exists():
                    print(f"Visualizing: {nc_file}")
                    visualize_soil_moisture(str(nc_file))
                else:
                    print(f"File not found: {nc_file}")
        else:
            if Path(file_path).exists():
                print(f"Visualizing: {file_path}")
                visualize_soil_moisture(str(file_path))
            else:
                print(f"File not found: {file_path}")
                print("Available files:")
                for file in Path(args.extract_dir).glob("*.nc"):
                    print(f"  - {file}")
    
    print("\n=== Complete ===")


if __name__ == "__main__":
    main()
        