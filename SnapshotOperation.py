import os
import time
import requests
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

# -------------------------------------------------------------------------
# Load environment variables
# -------------------------------------------------------------------------
load_dotenv()


# -------------------------------------------------------------------------
# 🧩 Helper: Monitor snapshot progress
# -------------------------------------------------------------------------
def check_snapshot_status(
    snapshot_id: str,
    max_retries: int = 60,
    interval: int = 5,
) -> bool:
    """
    Continuously poll the Bright Data API until a dataset snapshot is ready.

    Args:
        snapshot_id (str): The unique identifier of the snapshot.
        max_retries (int): Maximum number of polling attempts before timeout.
        interval (int): Time delay (in seconds) between checks.

    Returns:
        bool: True if the snapshot completes successfully, False otherwise.
    """
    api_key = os.getenv("BRIGHTDATA_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing BRIGHTDATA_API_KEY in environment variables.")

    progress_endpoint = f"https://api.brightdata.com/datasets/v3/progress/{snapshot_id}"
    headers = {"Authorization": f"Bearer {api_key}"}

    for attempt in range(1, max_retries + 1):
        try:
            print(f"⏳ Checking snapshot status... (Attempt {attempt}/{max_retries})")
            response = requests.get(progress_endpoint, headers=headers)
            response.raise_for_status()

            data = response.json()
            status = data.get("status")

            if status == "ready":
                print("✅ Snapshot is ready for download!")
                return True
            elif status == "failed":
                print("❌ Snapshot processing failed.")
                return False
            elif status == "running":
                print("🔄 Snapshot still in progress...")
            else:
                print(f"❓ Unexpected status received: {status}")

        except Exception as err:
            print(f"⚠️ Error while checking snapshot: {err}")

        time.sleep(interval)

    print("⏰ Timed out waiting for snapshot to complete.")
    return False


# -------------------------------------------------------------------------
# 📥 Helper: Download snapshot data
# -------------------------------------------------------------------------
def fetch_snapshot_data(
    snapshot_id: str,
    file_format: str = "json",
) -> Optional[List[Dict[Any, Any]]]:
    """
    Download and parse Bright Data snapshot results once ready.

    Args:
        snapshot_id (str): The unique identifier of the snapshot.
        file_format (str): Desired output format ('json', 'csv', etc.).

    Returns:
        Optional[List[Dict[Any, Any]]]: Parsed data list if successful, None otherwise.
    """
    api_key = os.getenv("BRIGHTDATA_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing BRIGHTDATA_API_KEY in environment variables.")

    download_endpoint = (
        f"https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}?format={file_format}"
    )
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        print("📦 Downloading snapshot content...")
        response = requests.get(download_endpoint, headers=headers)
        response.raise_for_status()

        data = response.json()
        item_count = len(data) if isinstance(data, list) else 1

        print(f"🎉 Download complete — retrieved {item_count} record(s).")
        return data

    except Exception as err:
        print(f"❌ Failed to download snapshot: {err}")
        return None