import csv
import os
import subprocess
from dataclasses import dataclass
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Dict, List

import ipadic
import pykakasi


class MeCabPos(Enum):
    NOUN = "名詞"
    NOUN_GENERAL = "一般"
    NOUN_NAMED_ENTITY = "固有名詞"
    NOUN_NUMBER = "数"


@dataclass
class MeCabItem(object):
    """MeCabItem to register to the user dictionary"""

    word: str
    tag: MeCabPos = MeCabPos.NOUN
    pos1: MeCabPos = MeCabPos.NOUN_GENERAL
    pos2: str = "*"
    pos3: str = "*"
    cost: int = 10

    def to_list(self) -> List[str]:
        kks = pykakasi.kakasi()
        kana = "".join([token["kana"] for token in kks.convert(self.word)])
        return [
            self.word.strip(),
            "*",
            "*",
            str(self.cost),
            self.tag.value,
            self.pos1.value,
            self.pos2,
            self.pos3,
            "*",
            "*",
            self.word.strip(),
            kana,
            kana,
        ]


class MeCabUtil(object):
    TEMP_DICT_CSV = Path("/tmp/tmp_dict.csv")

    @classmethod
    def add_noun_to_dict(cls, items: List[MeCabItem], user_dic: PathLike = Path("/usr/local/lib/mecab/user.dic")):
        """add a new word to MeCab user dictionary

        Args:
            items (List[MeCabItem]): MeCabItems to add to the dictionary.
        """
        words: Dict[str, List[str]] = {}

        # check the dictionary if the word is already added
        if cls.TEMP_DICT_CSV.exists():
            with open(cls.TEMP_DICT_CSV, mode="r", encoding="utf-8") as f:
                reader = csv.reader(f)
                words = {line[0]: line for line in reader}

        # register new word
        for item in items:
            if item.word in words:
                if item.pos2 != "*":
                    words[item.word][6] = item.pos2
                if item.pos3 != "*":
                    words[item.word][7] = item.pos3
                if item.cost != 10:
                    words[item.word][3] = str(item.cost)
            else:
                words[item.word] = item.to_list()

        # write into the user dictionary csv
        with open(cls.TEMP_DICT_CSV, mode="w", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(words.values())

        # update the mecab dictionary
        mecab_dict_index = Path("/usr/lib/mecab/mecab-dict-index")
        assert mecab_dict_index.exists(), f"No Such File: {str(mecab_dict_index.absolute())}"

        Path(user_dic).parent.mkdir(exist_ok=True, parents=True)

        subprocess.run(
            f"{str(mecab_dict_index.absolute())} "
            + f"-d {ipadic.DICDIR} "
            + f"-u {str(user_dic)} -f utf-8 -t utf-8 {str(cls.TEMP_DICT_CSV)}",
            shell=True,
        )

        print(f"compiled user dictionary -> {user_dic}")
