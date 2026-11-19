from setuptools import find_packages, setup

setup(
    name="bxt",
    packages=find_packages(exclude=["tests"]),
    version="0.1.0",
    description="Bayesian expected threat (xT) for football, IFT6269 (UdeM, Fall 2026)",
    author="[FILL IN TEAM MEMBERS]",
    license="",
    python_requires=">=3.9",
)
