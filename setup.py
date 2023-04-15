from __future__ import annotations

import pathlib

from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

setup(
    name="MA_minigrid", 
    version="0.0.2", 
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    author="Yang Gu",
    adescription="Multi-Agent Minigrid environment for reinforcement learning",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.10",
    ],
    install_requires=[
        "numpy>=1.18.0",
        "gymnasium>=0.26",
        "pygame>=2.2.0",
        "minigrid>=2.2.1",
    ],
)