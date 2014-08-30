from ez_setup import use_setuptools
use_setuptools()
from setuptools import setup,find_packages

setup(
    name='discomll',
    version='0.1.0',
    author='Roman Orac',
    author_email='orac.roman@gmail.com',
    packages = find_packages(), # adding packages
    package_data = {"": ["*.txt", "*.sh"]}, #add files from all packages
    test_suite = "tests",
    license = 'Apache License, Version 2.0',
    description = "DiscoMLL is a Python module for machine learning with MapReduce.",
    long_description = open('README.md').read(),
    zip_safe = False,
)