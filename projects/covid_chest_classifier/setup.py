from setuptools import setup, find_packages

setup(
    name="covid_chest_classifier",
    version="0.1",
    packages=find_packages(),  # automatically finds src and subpackages
    install_requires=[
        "torch",
        "pandas",
        "tqdm",
    ],
)
