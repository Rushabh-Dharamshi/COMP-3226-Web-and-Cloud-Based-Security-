import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# The directory containing the logs
base_url = "https://www.secrepo.com/self.logs/"
download_folder = "secrepo_logs"

# Create folder if it doesn't exist
if not os.path.exists(download_folder):
    os.makedirs(download_folder)

print(f"Scanning {base_url} for log files...")

# Get the directory page
response = requests.get(base_url)
soup = BeautifulSoup(response.text, "html.parser")

# Find all links that end in .gz
links = soup.find_all('a')
log_files = [link.get('href') for link in links if link.get('href').endswith('.gz')]

print(f"Found {len(log_files)} files. Starting download...")

for filename in log_files:
    # Construct full URL
    file_url = urljoin(base_url, filename)
    local_path = os.path.join(download_folder, filename)
    
    print(f"Downloading {filename}...")
    
    # Stream the download to avoid using too much RAM
    with requests.get(file_url, stream=True) as r:
        r.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

print("All downloads complete!")