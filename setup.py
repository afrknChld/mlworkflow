from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="mlworkflow",
    author="Istasse M.",
    author_email="istassem@gmail.com",
    version="0.0.1",
    python_requires='>=3.6',
    description="A workflow-improving library for manipulating ML experiments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=("mlworkflow",)),
    install_requires=["numpy", "tqdm", "ipywidgets"]
)
