from setuptools import setup, find_packages, Extension

setup(
    name="metastable",
    version="1.0.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=["qutip", "numpy", "scipy", "sympy", "tqdm"],
)
