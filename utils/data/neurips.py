from pathlib import Path
import json
from utils.google_drive import GDriveObjects, download_from_google_drive

def download_neurips_2021() -> dict:
    cache_path = Path('__cache__/datasets/neurips_2021.json')
    if not cache_path.exists():
        download_from_google_drive(GDriveObjects.NeurIPS_2021.value, str(cache_path))
    data = json.load(open(cache_path))
    return data