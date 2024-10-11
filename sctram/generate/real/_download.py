#!/usr/bin/env python3

import logging
import tempfile
from pathlib import Path
from random import choice
from string import ascii_lowercase
from typing import Optional, Union
from urllib.parse import parse_qs, urlparse
from zipfile import BadZipFile, ZipFile

import gdown
import requests
from filelock import SoftFileLock, Timeout
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("downloader")


def is_google_drive_url(url: str) -> bool:
    """Checks if a URL is a Google Drive share link.

    Args:
        url (str): The URL to check.

    Returns:
        bool: True if it's a Google Drive URL, False otherwise.
    """
    parsed_url = urlparse(url)
    return "drive.google.com" in parsed_url.netloc


def extract_google_drive_file_id(url: str) -> Optional[str]:
    """Extracts the Google Drive file ID from a share URL.

    Args:
        url (str): The Google Drive share URL.

    Returns:
        Optional[str]: The file ID if found, else None.
    """
    parsed_url = urlparse(url)
    if parsed_url.path.startswith("/file/d/"):
        # e.g., https://drive.google.com/file/d/<id>/view?usp=share_link
        parts = parsed_url.path.split("/")
        if len(parts) >= 4:
            return parts[3]
    else:
        # Parse query parameters
        query_params = parse_qs(parsed_url.query)
        if "id" in query_params:
            return query_params["id"][0]
    return None


def download_dataset(
    url: str,
    filename: Optional[str] = None,
    destination: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
    unzip: bool = False,
    timeout: int = 10,
) -> None:
    """Downloads a dataset from a given URL, supporting Google Drive links and direct URLs.

    Optionally unzips the file after download.

    Args:
        url (str): The URL to download the dataset from.
        filename (Optional[str]): Desired name for the downloaded file. If not provided, a random name is generated.
        destination (Union[str, Path], optional): Directory to save the downloaded file. Defaults to system temp directory.
        overwrite (bool): If True, overwrites the file if it already exists. Defaults to False.
        unzip (bool): If True and the file is a ZIP archive, extracts its contents. Defaults to False.
        timeout (int): Maximum time in seconds to wait for acquiring the file lock. Defaults to 10.
    """
    # Detect if the URL is a Google Drive URL
    is_gdrive = is_google_drive_url(url)

    # Generate a random filename if not provided
    if not filename:
        random_suffix = "".join(choice(ascii_lowercase) for _ in range(10))
        filename = f"dataset_{random_suffix}"

    # Set destination to temp directory if not specified
    if not destination:
        destination = Path(tempfile.gettempdir())
    else:
        destination = Path(destination)

    # Ensure the destination directory exists
    destination.mkdir(parents=True, exist_ok=True)

    # Full path for the downloaded file
    download_path = destination / filename

    # Lock file path to prevent concurrent downloads
    lock_path = download_path.with_suffix(download_path.suffix + ".lock")

    # Initialize the lock
    lock = SoftFileLock(lock_path, timeout=timeout)

    try:
        with lock:
            if download_path.exists() and not overwrite:
                logger.warning(f"File {download_path!r} already exists. Skipping download.")
                return

            # Temporary file path for streaming download
            temp_download_path = download_path.with_suffix(".part")

            if is_gdrive:
                logger.info("Detected Google Drive URL. Using gdown for download.")
                # Extract file ID
                file_id = extract_google_drive_file_id(url)
                if not file_id:
                    logger.error("Failed to extract Google Drive file ID. Aborting download.")
                    return
                # Generate a direct download URL compatible with gdown
                direct_download_url = f"https://drive.google.com/uc?id={file_id}"

                try:
                    logger.debug(f"Starting Google Drive download with gdown. URL: {direct_download_url}")
                    # gdown handles the confirmation token internally
                    # fuzzy=True allows gdown to accept various Google Drive URL formats
                    gdown.download(direct_download_url, str(temp_download_path), quiet=False, fuzzy=True)
                except Exception as e:
                    logger.error(f"gdown failed to download the file. Error: {e}")
                    if temp_download_path.exists():
                        temp_download_path.unlink()  # Remove incomplete download
                    return
            else:
                # Use requests for direct downloads
                try:
                    logger.info(f"Starting download from URL: {url}")
                    with requests.get(url, stream=True) as response:
                        response.raise_for_status()
                        total_size = int(response.headers.get("content-length", 0))
                        logger.debug(f"Total file size: {total_size} bytes")

                        # Initialize tqdm progress bar
                        with tqdm(total=total_size, unit="B", unit_scale=True, desc=f"Downloading {filename}") as pbar:
                            with temp_download_path.open("wb") as f:
                                for chunk in response.iter_content(chunk_size=8192):
                                    if chunk:  # Filter out keep-alive chunks
                                        f.write(chunk)
                                        pbar.update(len(chunk))
                except requests.RequestException as e:
                    logger.error(f"Failed to download the file. Error: {e}")
                    if temp_download_path.exists():
                        temp_download_path.unlink()  # Remove incomplete download
                    return

            # Rename the temporary file to the final download path
            temp_download_path.rename(download_path)
            logger.info(f"Downloaded {download_path!r} successfully.")

            # If the file is a ZIP archive and unzip is requested
            if unzip and download_path.suffix.lower() == ".zip":
                try:
                    with ZipFile(download_path, "r") as zip_ref:
                        zip_ref.extractall(destination)
                    logger.info(f"Extracted {download_path!r} to {destination!r}.")
                    # Optionally, remove the ZIP file after extraction
                    # download_path.unlink()
                except BadZipFile:
                    logger.error(f"The file {download_path!r} is not a valid ZIP archive.")

    except Timeout:
        logger.error(f"Failed to acquire lock on {lock_path!r}. Another download might be in progress.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
    finally:
        if lock.is_locked:
            lock.release()

    # Check if the lock file still exists and handle it
    if lock_path.exists():
        try:
            lock_path.unlink()
            logger.info(f"Removed stale lock file {lock_path!r}.")
        except Exception as e:
            logger.warning(f"Failed to remove stale lock file {lock_path!r}. Error: {e}")
