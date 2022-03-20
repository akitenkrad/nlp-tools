from pathlib import Path
import requests
from enum import Enum

def download_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = __get_confirm_token(response)
    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)
    __save_google_drive_content(response, destination)    

def __get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def __save_google_drive_content(response, destination):
    CHUNK_SIZE = 32768
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

class GDriveObjects(Enum):
    NeurIPS_2021 = '1JmYpSatyr2OTCPpovUDQa3PzJmz39USV'
