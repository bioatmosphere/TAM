#!/bin/bash

# Configuration
BASE_URL="https://climate.northwestknowledge.net/TERRACLIMATE-DATA"
START_YEAR=2001
END_YEAR=2010
OUTPUT_DIR="../ancillary/terraclimate"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Download function with error handling
download_file() {
    local year=$1
    local file="TerraClimate_ppt_${year}.nc"
    local url="${BASE_URL}/${file}"
    
    echo "Downloading data for year ${year}..."
    wget -nc -c -nd --directory-prefix="$OUTPUT_DIR" "$url" || {
        echo "Error downloading $file"
        return 1
    }
}

# Main download loop
for year in $(seq $START_YEAR $END_YEAR); do
    download_file $year
done

echo "Download completed. Files saved in $OUTPUT_DIR"
