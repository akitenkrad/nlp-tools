![GitHub](https://img.shields.io/github/license/akitenkrad/nlp-tools?style=for-the-badge)
![GitHub last commit](https://img.shields.io/github/last-commit/akitenkrad/nlp-tools?style=for-the-badge)
![Code Style](https://img.shields.io/badge/code%20style-black-black?style=for-the-badge)
![Code Style](https://img.shields.io/badge/code%20style-flake8-black?style=for-the-badge)
![Code Style](https://img.shields.io/badge/imports-isort-blue?style=for-the-badge)
![Code Style](https://img.shields.io/badge/typing-mypy-blue?style=for-the-badge)

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