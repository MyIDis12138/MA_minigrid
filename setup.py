from __future__ import annotations

import pathlib

from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

setup(
    name="MA_minigrid", 
    version="0.1.0", 
    packages=find_packages(exclude=["**/__pycache__"]),
    author="Yang Gu",
    adescription="Multi-Agent Minigrid environment for reinforcement learning",
    classifiers=[
        "Development Status :: 1 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.10",
    ],
    install_requires=[
        "numpy>=1.18.0",
        "gymnasium>=0.26",
        "pygame>=2.2.0",
        "minigrid>=2.2.0",
        'openai',
    ],
)