import json
import os
from pathlib import Path


def check_kaggle_configure() -> bool:
    cfg_path = Path("~/.kaggle/kaggle.json").expanduser().absolute()
    if not cfg_path.exists():
        return False
    cfg = json.load(open(cfg_path))
    if "username" not in cfg or "key" not in cfg:
        return False
    if len(cfg["username"]) <= 0 or len(cfg["key"]) <= 0:
        return False
    return True


def kaggle_configure():
    print("Kaggle Configure:")

    print("username:", end="")
    username = input()

    print("key:", end="")
    key = input()

    kaggle_config = {"username": username, "key": key}
    outpath = Path("~/.kaggle/kaggle.json").expanduser().absolute()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, mode="wt", encoding="utf-8") as wf:
        os.chmod(outpath, 0o600)
        json.dump(kaggle_config, wf, ensure_ascii=False)

    print("Done.")
