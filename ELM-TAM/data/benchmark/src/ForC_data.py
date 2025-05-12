"""
Downloads the data folder from the ForC GitHub repository at a specific commit.

Reference:
https://g.co/gemini/share/e76a175e6819
"""


import requests
import zipfile
import io
import os

# --- Configuration ---
USER = "forc-db"
REPO = "ForC"
COMMIT_HASH = "407c520e6350917bca42e6bf7d5031dbcc551362"
FOLDER_PATH_IN_REPO = "data" # The specific folder you want to download
LOCAL_DESTINATION_DIR = "../productivity/forc/data" # Where to save the folder contents locally
# --- End Configuration ---

# Construct the download URL for the zip archive of the specific commit
zip_url = f"https://github.com/{USER}/{REPO}/archive/{COMMIT_HASH}.zip"

# The expected path prefix for the desired folder within the zip file.
# GitHub zip archives typically contain a top-level directory named <repo>-<commit_hash>
zip_folder_prefix = f"{REPO}-{COMMIT_HASH}/{FOLDER_PATH_IN_REPO}/"

print(f"Attempting to download folder '{FOLDER_PATH_IN_REPO}'")
print(f"From: github.com/{USER}/{REPO}")
print(f"Commit: {COMMIT_HASH}")
print(f"Download URL: {zip_url}")
print("-" * 20)

try:
    # 1. Download the zip file content
    response = requests.get(zip_url, stream=True)
    response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
    print("Download successful. Preparing to extract...")

    # 2. Use BytesIO to treat the downloaded content as an in-memory file
    zip_content = io.BytesIO(response.content)

    # 3. Process the zip file
    with zipfile.ZipFile(zip_content, 'r') as zip_ref:
        # Create the local destination directory if it doesn't exist
        os.makedirs(LOCAL_DESTINATION_DIR, exist_ok=True)
        print(f"Extracting contents of '{FOLDER_PATH_IN_REPO}' to: {os.path.abspath(LOCAL_DESTINATION_DIR)}")

        extracted_count = 0
        # Iterate through each file/directory in the zip archive
        for member in zip_ref.infolist():
            # Check if the member is within the target folder we want
            # We also ensure it's not the folder entry itself (which ends with '/')
            # but rather the files *inside* it or subfolders.
            if member.filename.startswith(zip_folder_prefix) and not member.filename == zip_folder_prefix:

                # Calculate the target path relative to the local destination directory
                # Example: "ForC-407c.../data/sub/file.txt" -> "sub/file.txt"
                relative_path = member.filename.split(zip_folder_prefix, 1)[1]
                target_path = os.path.join(LOCAL_DESTINATION_DIR, relative_path)

                # Ensure the target directory exists (needed for files in subdirectories)
                if member.is_dir():
                    os.makedirs(target_path, exist_ok=True)
                    # print(f"  Creating directory: {target_path}") # Uncomment for verbose output
                else:
                    # Create parent directory for the file if it doesn't exist
                    parent_dir = os.path.dirname(target_path)
                    os.makedirs(parent_dir, exist_ok=True)

                    # Extract the file by reading from zip and writing locally
                    with zip_ref.open(member) as source, open(target_path, "wb") as target:
                        target.write(source.read())
                    # print(f"  Extracting file: {target_path}") # Uncomment for verbose output
                    extracted_count += 1

        if extracted_count > 0:
            print(f"\nSuccessfully extracted {extracted_count} file(s)/folder(s) from '{FOLDER_PATH_IN_REPO}'.")
        else:
            print(f"\nWarning: No files found within the specified folder '{FOLDER_PATH_IN_REPO}' in the archive.")
            print(f"Expected prefix in zip: '{zip_folder_prefix}'")


except requests.exceptions.RequestException as e:
    print(f"\nError during download: {e}")
    print("Please check the URL, commit hash, and your internet connection.")
except zipfile.BadZipFile:
    print("\nError: Downloaded file is not a valid zip archive.")
except KeyboardInterrupt:
    print("\nDownload/Extraction cancelled by user.")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")