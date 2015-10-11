from setuptools import setup,find_packages

setup(
    name='discomll',
    version='0.1.4',
    author='Roman Orac',
    author_email='orac.roman@gmail.com',
    url="https://github.com/romanorac/discomll",
    packages = find_packages(), # adding packages
    package_data = {"": ["*.txt", "*.sh", "*.csv"]}, #add files from all packages
    test_suite = "tests",
    license = 'Apache License, Version 2.0',
    description = "Disco Machine Learning Library."
)