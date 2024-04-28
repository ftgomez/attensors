import os

from setuptools import find_packages, setup

project_root = os.path.dirname(os.path.abspath(__file__))
os.environ["PYTHONPATH"] = project_root + ":" + os.environ.get("PYTHONPATH", "")


setup(
    name="attensors",
    version="1.0",
    packages=find_packages(),
    install_requires=["torch==2.2.2", "numpy==1.26.4"],
)
