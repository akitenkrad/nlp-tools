![GitHub](https://img.shields.io/github/license/akitenkrad/nlp-tools?style=for-the-badge)
![GitHub last commit](https://img.shields.io/github/last-commit/akitenkrad/nlp-tools?style=for-the-badge)

![GitHub Pipenv locked dependency version](https://img.shields.io/github/pipenv/locked/dependency-version/akitenkrad/nlp-tools/torch?style=for-the-badge)
![GitHub Pipenv locked dependency version](https://img.shields.io/github/pipenv/locked/dependency-version/akitenkrad/nlp-tools/transformers?style=for-the-badge)
![GitHub Pipenv locked dependency version](https://img.shields.io/github/pipenv/locked/dependency-version/akitenkrad/nlp-tools/bertopic?style=for-the-badge)
![GitHub Pipenv locked dependency version](https://img.shields.io/github/pipenv/locked/dependency-version/akitenkrad/nlp-tools/wordcloud?style=for-the-badge)
![GitHub Pipenv locked dependency version](https://img.shields.io/github/pipenv/locked/dependency-version/akitenkrad/nlp-tools/mecab-python3?style=for-the-badge)


# NLP-Tools

## Datasets

### Conference Datasets

```python
from datasets.conference import NeurIPS_2021
texts = NeurIPS_2021.load()
```

Available datasets.
- NeurIPS 2021  
- ANLP 2022
- JSAI 2022

## Development

### Setup Environment

1. Create Docker container
```bash
$ docker-compose up -d
```

2. Install dependencies
```bash
$ pipenv install --python 3.9 --dev
```

3. Enter virtual environment
```bash
$ pipenv shell
```

### Test

1. Write tests.
```
tests/
└── utils
    └── test_tokenizers.py
```

2. Run the `pytest` command.

```bash
$ pytest
======= 4 passed, in 29.43s ======
```