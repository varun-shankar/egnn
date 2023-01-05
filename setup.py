from setuptools import setup, find_packages
from pathlib import Path

# see https://packaging.python.org/guides/single-sourcing-package-version/
# version_dict = {}
# with open(Path(__file__).parents[0] / "nequip/_version.py") as fp:
#     exec(fp.read(), version_dict)
# version = version_dict["__version__"]
# del version_dict

setup(
    name="egnn",
    description="Equivariant GNN",
    author="Varun Shankar",
    python_requires=">=3.8",
    packages=find_packages(include=["egnn", "egnn.*"]),
    install_requires=[
        "numpy",
        "torch",
        "e3nn",
        "setuptools",
        "pytorch-lightning",
        "wandb"
    ],
    zip_safe=True,
)