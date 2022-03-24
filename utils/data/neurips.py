from typing import List
from pathlib import Path
import json
from utils.google_drive import GDriveObjects, download_from_google_drive
from stats.stats import Text

def download_neurips_2021() -> List[Text]:
    cache_path = Path('__cache__/datasets/neurips_2021.json')
    if not cache_path.exists():
        download_from_google_drive(GDriveObjects.NeurIPS_2021.value, str(cache_path))
    json_data = json.load(open(cache_path))
    data = []
    for paper in json_data:
        text = Text(title=paper['title'], summary=paper['abstract'],
                    keywords=paper['keywords'], pdf_url=paper['pdf_url'],
                    authors=paper['authors'],
                    **{
                        'url': paper['url'],
                        'authors': paper['authors'],
                        'grade': paper['grade'],
                        'date': paper['date']
                    }
        )
        data.append(text)
    return data
