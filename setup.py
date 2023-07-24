import sys

import setuptools

with open("README.md", mode="r") as f:
    long_description = f.read()

version_range_max = max(sys.version_info[1], 10) + 1
python_min_version = (3, 8, 0)

setuptools.setup(
    name="nlp_tools",
    version="0.0.1",
    author="akitenkrad",
    author_email="akitenkrad@gmail.com",
    packages=setuptools.find_packages(),
    package_data={
        "nlp_tools": [
            "utils/resources/mask_images/*.png",
            "utils/resources/texts/*.txt",
            "config/*.yml",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
    ]
    + ["Programming Language :: Python :: 3.{}".format(i) for i in range(python_min_version[1], version_range_max)],
    long_description=long_description,
    install_requires=[
        "arxiv",
        "beautifulsoup4",
        "click",
        "colorama",
        "gensim",
        "attrdict @ git+https://github.com/akitenkrad/attrdict",
        "h5py",
        "ipadic",
        "ipython",
        "ipywidgets",
        "kaggle",
        "kaleido",
        "mecab-python3",
        "mlflow",
        "networkx",
        "nltk",
        "numpy",
        "pandas",
        "patool",
        "plotly",
        "progressbar",
        "py-cpuinfo",
        "pykakasi",
        "python-dateutil",
        "python-dotenv",
        "pyunpack",
        "PyYAML",
        "requests",
        "scikit-learn",
        "scipy",
        "seaborn",
        "sumeval",
        "tensorboard",
        "torch",
        "torchinfo",
        "torchtext",
        "torchvision",
        "tqdm",
        "unidic",
        "wordcloud",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "mypy",
            "flake8",
            "isort",
            "jupyterlab",
            "types-python-dateutil",
            "types-PyYAML",
            "types-requests",
            "typing-extensions",
        ]
    },
)
