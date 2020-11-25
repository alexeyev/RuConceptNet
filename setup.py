# -*- coding: utf-8 -*-

from distutils.core import setup

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ruconceptnet",
    packages=setuptools.find_packages(),
    version="0.0.1",
    description="Russian part of ConceptNet5.7 graph, fast access to relations.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Anton Alekseev",
    author_email="anton.m.alexeye@gmail.com",
    url="https://github.com/alexeyev/RussianConceptNet",
    keywords=["nlp", "knowledge graph", "russian language"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    zip_safe=False,
    include_package_data=True,
    python_requires=">=3.5",
)
