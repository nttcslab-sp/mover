from distutils.extension import Extension
import numpy

from setuptools import setup, find_packages

from Cython.Build import cythonize

ext_modules = cythonize(
    [
        Extension(
            "mover._meeteval.wer.matching.cy_levenshtein",
            ["mover/_meeteval/wer/matching/cy_levenshtein.pyx"],
            extra_compile_args=["-std=c++11"],
            extra_link_args=["-std=c++11"],
        ),
    ]
)

# Get the long description from the relevant file
try:
    from os import path

    here = path.abspath(path.dirname(__file__))
    with open(path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = ""

setup(
    name="mover",
    # Versions should comply with PEP440.
    version="0.0.0",
    # The project's main homepage.
    url="https://github.com/nttcslab-sp/mover",
    # Choose your license
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 5 - Production/Stable",
        # Indicate who your project is intended for
        "Intended Audience :: Science/Research",
        # Pick your license as you wish (should match "license" above)
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.5",
    author="",
    author_email="",
    keywords="speech recognition, word error rate, evaluation, meeting, ASR, WER",
    long_description=long_description,
    long_description_content_type="text/markdown",  # Optional (see note above)
    ext_modules=ext_modules,
    packages=find_packages(exclude=["contrib", "docs", "tests*"]),
    setup_requires=[
        "numpy",
        "typing_extensions; python_version<'3.8'",  # Missing Literal in py37
        "cached_property; python_version<'3.8'",  # Missing functools.cached_property in py37
        "Cython",
    ],
    install_requires=[
        "dover-lap",
        "meeteval",
        "simplejson",
    ],
    extras_require={"test": ["pytest", "pytest-cov"]},
    package_data={
        "mover": ["**/*.pyx", "**/*.h", "**/*.js", "**/*.css", "**/*.html"]
    },  # https://stackoverflow.com/a/60751886
    entry_points={"console_scripts": ["mover=mover.__main__:cli"]},
    include_dirs=[numpy.get_include()],
)
