from setuptools import setup, find_packages

setup(
    name="distml-platform",
    version="1.0.0",
    author="BandageOK",
    description="Enterprise-grade distributed machine learning training platform",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
    ],
)
