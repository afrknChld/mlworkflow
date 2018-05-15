from setuptools import setup, find_packages
from os import path


setup(
    name="mlworkflow",
    author="Istasse M.",
    author_email="istassem@gmail.com",
    license='MIT',
    version="0.0.3",
    python_requires='>=3.6',
    description="A workflow-improving library for manipulating ML experiments",
    long_description_content_type="text/markdown",
    packages=find_packages(include=("mlworkflow",)),
    install_requires=["numpy", "tqdm", "ipywidgets"]
)
