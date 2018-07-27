from setuptools import setup, find_packages
from os import path


setup(
    name="mlworkflow",
    author="Maxime Istasse",
    author_email="istassem@gmail.com",
    url="https://github.com/mistasse/mlworkflow",
    license='MIT',
    version="0.0.5",
    python_requires='>=3.6',
    description="A workflow-improving library for manipulating ML experiments",
    long_description_content_type="text/markdown",
    packages=find_packages(include=("mlworkflow",)),
    install_requires=["numpy", "tqdm", "ipywidgets"],
)
