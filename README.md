# NLP-Tools

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