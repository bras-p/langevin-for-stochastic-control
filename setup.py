from setuptools import setup, find_packages

setup(
    name='langevin_optimizers',
    packages=[
        "langevin_optimizers",
    ],
    url='https://github.com/Bras-P/langevin-for-stochastic-control',
    author="Pierre Bras",
    description='An implementation of Langevin optimizers in TensorFlow',
    long_description=open('README.md').read(),
    install_requires=[
        "tensorflow>=2.10.0",
        ],
    include_package_data=True,
    license='MIT',
)
