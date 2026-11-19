from setuptools import find_packages, setup

setup(
    name="pgm",
    packages=find_packages(exclude=["tests"]),
    version="0.1.0",
    description="Bayesian expected threat (xT) for football",
    author="[Ryan Kahkeshan, Abdoul Karim Ouattara, Xavier Drolet, Pierre Emery]",
    license="",
    python_requires=">=3.9",
)
