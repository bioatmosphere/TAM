import pandas as pd
from chardet import detect


import requests
from pathlib import Path
import os
from tqdm import tqdm
from zipfile import ZipFile
from glob import glob

def save_webpage(url, output_file):
    """
    Save a webpage as an HTML file.
    
    Args:
        url (str): URL of the webpage to save
        output_file (str): Path where the HTML file will be saved
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Send a GET request to the URL
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Save the content to a file
        with open(output_file, 'wb') as file:
            file.write(response.content)
        
        print(f"Webpage saved successfully to {output_file}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the webpage: {e}")
        return False
    except IOError as e:
        print(f"Error saving the file: {e}")
        return False
    
def download_file(url, output_path):
    """
    Download a file from URL with progress bar and error handling.
    
    Args:
        url (str): URL to download from
        output_path (str): Path where to save the file
    """
    try:
        # Send a GET request to the URL
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Get total file size
        total_size = int(response.headers.get('content-length', 0))
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Download with progress bar
        with open(output_path, 'wb') as file, tqdm(
            desc=os.path.basename(output_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                pbar.update(size)
                
        print(f"Download completed: {output_path}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return False

# Download the source data in CSV format
url = "https://datadryad.org/dataset/doi:10.5061/dryad.7sqv9s4vv"
download_url = "http://datadryad.org/api/v2/datasets/doi%253A10.5061%252Fdryad.7sqv9s4vv/download"

output_file = 'grassland_npp_data.csv'
output_file_path = "../productivity/grassland/source_data.zip"

# Save the webpage
#success = save_webpage(url, output_file)

# Download the file
success = download_file(download_url, output_file_path)

# Unzip just the monthly data
for zipfile in tqdm(glob("../productivity/grassland/*.zip"), desc="Unzipping"):
    #csvfile = Path(zipfile.replace("SUBSET", "SUBSET_MM").replace("zip", "csv"))
    with ZipFile(zipfile) as fzip:
        # if csvfile.is_file():
        #     continue
        if (fzip.filename == 'Data_S1.zip'):
            fzip.extractall(path="../productivity/grassland/")
            continue
        else:
            fzip.extractall(path="../productivity/grassland/")

# # Check the encoding of the CSV file
directory_path = "../productivity/grassland/Data_S1/"
csv_files = glob(f"{directory_path}/*.csv")
print(f"CSV files found: {csv_files}")

with open(csv_files[0], "rb") as f:
    result = detect(f.read())
    encoding = result['encoding']
    print(f"Detected encoding: {encoding}")

# # Read the CSV file with the detected encoding
# df = pd.read_csv(file_path, encoding=encoding)