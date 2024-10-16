import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin, urlparse


def is_file(url):
    """
    Check if the URL points to a downloadable file by checking the content type in the headers.
    """
    try:
        response = requests.head(url, allow_redirects=True)
        content_type = response.headers.get('Content-Type', '')
        # Check if content type indicates a file (like 'application/octet-stream', 'image/jpeg', etc.)
        if 'text/html' not in content_type:
            return True
    except requests.RequestException as e:
        print(f"Error checking URL {url}: {e}")
    return False


def download_file(url, save_dir="downloads"):
    """
    Download the file from the URL and save it to a directory.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    local_filename = os.path.join(save_dir, os.path.basename(urlparse(url).path))

    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Downloaded: {local_filename}")
    except requests.RequestException as e:
        print(f"Failed to download {url}: {e}")


def get_links(url):
    """
    Get all the links from the given webpage URL.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Ensure the request was successful
        soup = BeautifulSoup(response.text, 'html.parser')
        links = [urljoin(url, a['href']) for a in soup.find_all('a', href=True)]
        return links
    except requests.RequestException as e:
        print(f"Error visiting URL {url}: {e}")
    return []


def process_url(url, visited=set()):
    """
    Process a URL, checking if it's a file, and if not, retrieving links and repeating the process.
    """
    if url in visited:
        return
    visited.add(url)

    # Check if the URL is a file
    if is_file(url):
        download_file(url)
    else:
        # It's a webpage, get all links and recursively process them
        links = get_links(url)
        for link in links:
            process_url(link, visited)


# Start the script with an initial URL
if __name__ == "__main__":
    start_url = "http://cycpeptmpdb.com/peptides/id_1/"  # Replace with the starting URL
    process_url(start_url)
