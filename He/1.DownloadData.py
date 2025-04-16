import requests
import os
import re

# Base URL for the Figshare API
BASE_API_URL = "https://api.figshare.com/v2/articles/13123148/files"

# Directory to save the downloaded files
DOWNLOAD_DIR = "He_Dataset"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def get_file_list():
    print("[INFO] Fetching full file list from Figshare with pagination...")
    all_files = []
    page = 1
    page_size = 100  # Max allowed by Figshare

    while True:
        paged_url = f"{BASE_API_URL}?page={page}&page_size={page_size}"
        response = requests.get(paged_url)
        response.raise_for_status()
        files = response.json()

        if not files:
            break

        all_files.extend(files)
        print(f"[INFO] Retrieved {len(files)} files from page {page}")
        page += 1

    print(f"[INFO] Total files retrieved: {len(all_files)}")
    return all_files


# Function to download a file given its download URL and name
def download_file(file_url, file_name):
    response = requests.get(file_url, stream=True)
    response.raise_for_status()
    file_path = os.path.join(DOWNLOAD_DIR, file_name)
    with open(file_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded: {file_name}")

# Main function to orchestrate the downloading process
def main():
    # Retrieve the list of files from Figshare
    files = get_file_list()
    print(files)
    # Compile a regex pattern to match desired files (Subjects 1–25, Sessions 1–4)
    #pattern = re.compile(r"S([2-9]|1[0-9]|2[0-5])_Session_[1-4]\.mat$")
    pattern = re.compile(r"S(1[9]|2[0-5])_Session_[1-4]\.mat$")
    
    # Filter and download files that match the pattern
    for file in files:
        file_name = file['name']
        if pattern.match(file_name):
            file_url = file['download_url']
            download_file(file_url, file_name)

if __name__ == "__main__":
    main()
