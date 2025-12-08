import os
import sys
import time
import subprocess
import requests
from pathlib import Path

# Github repo information
REPO_OWNER = "Dillylol"
REPO_NAME = "client"
GITHUB_API_URL = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/releases/latest"

def check_for_updates(current_version):
    """
    Checks GitHub for the latest release.
    Returns the download URL if a newer version is available, else None.
    """
    print(f"Checking for updates... (Current version: {current_version})")
    try:
        response = requests.get(GITHUB_API_URL, timeout=5)
        response.raise_for_status()
        latest_release = response.json()
        
        latest_version = latest_release["tag_name"]
        # Normalize versions (remove 'v' prefix if present)
        current_v = current_version.lstrip("v")
        latest_v = latest_version.lstrip("v")

        print(f"Latest version on GitHub: {latest_version}")

        # Simple string comparison check (assumes semantic versioning isn't strictly needed if strict equality fails)
        # Ideally we'd parse this, but for now exact match '!=' is safe enough to trigger update if they differ.
        if latest_v != current_v:
             # Find the executable asset
            for asset in latest_release.get("assets", []):
                # Assuming the asset name contains the app name or is the only exe
                if asset["name"].endswith(".exe"):
                    return asset["browser_download_url"]
            print("New release found, but no .exe asset detected.")
    except Exception as e:
        print(f"Failed to check for updates: {e}")
    
    return None

def update_application(download_url):
    """
    Downloads the new executable and replaces the current one using a batch script.
    """
    print(f"Downloading update from {download_url}...")
    
    # Current executable path
    if getattr(sys, 'frozen', False):
        current_exe = sys.executable
    else:
        print("Not running as frozen executable. Skipping update.")
        return False

    current_dir = os.path.dirname(current_exe)
    new_exe_name = "DevController_new.exe"
    new_exe_path = os.path.join(current_dir, new_exe_name)
    
    try:
        # Download the new file
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        with open(new_exe_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print("Download complete. Preparing to restart...")

        # Create the update batch script
        batch_script = os.path.join(current_dir, "update.bat")
        exe_name = os.path.basename(current_exe)
        
        # Batch script:
        # 1. Wait a bit
        # 2. Delete old exe
        # 3. Rename new exe to old exe
        # 4. Start old exe (which is now the new one)
        # 5. Delete self
        batch_content = f"""
@echo off
timeout /t 2 /nobreak >nul
del "{exe_name}"
move "{new_exe_name}" "{exe_name}"
start "" "{exe_name}"
del "%~f0"
"""
        with open(batch_script, "w") as f:
            f.write(batch_content)

        # Run the batch script and exit
        subprocess.Popen([batch_script], shell=True)
        return True

    except Exception as e:
        print(f"Update failed: {e}")
        if os.path.exists(new_exe_path):
            os.remove(new_exe_path)
        return False

def check_and_update(current_version):
    """
    Wrapper function to check for updates and apply them.
    If update is successful, this function exits the application.
    """
    download_url = check_for_updates(current_version)
    if download_url:
        print(f"Update available! Downloading from {download_url}")
        if update_application(download_url):
            sys.exit(0)