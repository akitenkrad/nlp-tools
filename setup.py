"""
To install nlp-tools:
    python setup.py install
"""
import sys
from setuptools import setup

DESCRIPTION = "nlp-tools"

try:
    LONG_DESCRIPTION = open('README.md').read()
except:
    LONG_DESCRIPTION = DESCRIPTION

python_min_version = (3, 8, 0)
python_min_version_str = '.'.join(map(str, python_min_version))
version_range_max = max(sys.version_info[1], 10) + 1

setup(
    name="nlp-tools",
    version="0.0.1",
    author="akitenkrad",
    author_email="akitenkrad@gmail.com",
    packages=("nlp-tools",),
    url="https://github.com/akintekrad/nlp-tools",
    license="MIT License",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            "License :: OSI Approved :: MIT License",
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Software Development',
            'Topic :: Software Development :: Libraries',
            'Topic :: Software Development :: Libraries :: Python Modules',
            'Programming Language :: Python :: 3',
        ] + ['Programming Language :: Python :: 3.{}'.format(i) for i in range(python_min_version[1], version_range_max)],
    zip_safe=True,
)
