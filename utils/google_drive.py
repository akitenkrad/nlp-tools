from enum import Enum
from os import PathLike
from pathlib import Path

import requests

from utils.utils import is_notebook

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def download_from_google_drive(id: str, dst_filename: PathLike):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={"id": id}, stream=True, timeout=None)
    token = __get_confirm_token(response)
    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True, timeout=None)
    __save_google_drive_content(response, dst_filename)


def __get_confirm_token(response):
    if "virus scan warning" in response.text.lower():
        return "t"
    return None


def __save_google_drive_content(response, dst_filename):
    CHUNK_SIZE = 32768
    path = Path(dst_filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    total = int(len(response.content) / CHUNK_SIZE)
    with open(path, mode="wb") as f:
        for chunk in tqdm(response.iter_content(CHUNK_SIZE), total=total, desc=f"save {dst_filename} ..."):
            if chunk:
                f.write(chunk)


class GDriveObjects(Enum):

    # MS-MARCO
    MSMARCO_TRAIN_DATA = "1Ul_5hZ1znlkmjciAVkL-R6mTvN2ZPUcF"
    MSMARCO_TRAIN_ROUGE = "10JXIrBqvWenEdVkI3b0MPeddx0VHr_w-"
    MSMARCO_DEV_DATA = "1bkIL69AjznJfdPEjgGlxORq-i_uNGgC3"
    MSMARCO_DEV_ROUGE = "11L08_GFK5lj9607or9JaZ6WR0SlxYEo4"
    MSMARCO_EXP_DATA = "1tVtIYSs7YDn6lv-1hK5WrDvS-Sj-Et5u"

    # Conference Datasets
    NeurIPS_2021 = "1JmYpSatyr2OTCPpovUDQa3PzJmz39USV"
    ANLP_2022 = "1A3Kk_tpaPPAd543j5xApco89gHAnKMCF"
    JSAI_2022 = "1-93EeiCJp5vyXatK7H6uLSIyozLeUFIL"
    ACL_20222 = "1-PtEnglF1SNk38H-yDMg4chAQf7KqsJe"
