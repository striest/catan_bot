from setuptools import setup, find_packages
from setuptools import find_packages

setup(
    name='catanbot',
    version='0.0',
    packages=find_packages(),
    license='MIT License',
    install_requires=[
        'numpy>=1.16',
        'matplotlib>=3.1',
        'ray>=0.8.6',
        'pygame>=2.0.0.dev6'
    ],
    long_description=open('README.md').read(),
)
