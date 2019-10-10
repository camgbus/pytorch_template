from setuptools import setup, find_packages

setup(
    name='python_template',
    version='0.1',
    description='A Python project template',
    url='https://github.com/camgbus/python_template',
    keywords='python setuptools',
    packages=find_packages(include=['pt', 'pt.*']),
)