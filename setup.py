from setuptools import setup, find_packages

setup(
    name='langevin-stochastic-control',
    packages=[
        "models",
        "optimizers",
        "data",
    ],
    url='https://github.com/Bras-P/langevin-for-stochastic-control',
    author="Pierre Bras",
    description='A framework for comparing optimizers on Stochastic control problems, with an implementation of Langevin optimizers in TensorFlow',
    long_description=open('README.md').read(),
    install_requires=[
        "tensorflow>=2.10.0",
        "pandas>=1.4.0",
        "numpy>=1.20.0",
        "matplotlib>=3.1.0",
        ],
    include_package_data=True,
    license='MIT',
)
