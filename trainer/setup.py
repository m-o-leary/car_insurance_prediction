from pathlib import Path
from setuptools import find_packages, setup

# Read the requirements
source_root = Path(".")
with (source_root / "requirements.txt").open(encoding="utf8") as f:
    requirements = f.readlines()

setup(
    name='trainer_lib',
    packages=find_packages(),
    version='0.0.1',
    description='Car insurance EDA, cleaning, transformation and modelling pipeline',
    author="Martin O' Leary",
    license='',
    install_requires=requirements
)
