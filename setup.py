#!/usr/bin/env python
import os
import io
import re
from setuptools import setup, find_packages
import setuptools.command.develop 
import setuptools.command.install 

version = '0.0.3'

try:
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], 
        cwd=cwd).decode('ascii').strip()
    version += '+' + sha[:7]
except Exception:
    pass

def create_version_file():
    global version, cwd
    print('-- Building version ' + version)
    version_path = os.path.join(cwd, 'gluoncvth', 'version.py')
    with open(version_path, 'w') as f:
        f.write('"""This is gluoncvth version file."""\n')
        f.write("__version__ = '{}'\n".format(version))

# run test scrip after installation
class install(setuptools.command.install.install):
    def run(self):
        create_version_file()
        setuptools.command.install.install.run(self)

class develop(setuptools.command.develop.develop):
    def run(self):
        create_version_file()
        setuptools.command.develop.develop.run(self)

try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst')
except(IOError, ImportError):
    long_description = open('README.md').read()

requirements = [
    'numpy',
    'torch',
    'tqdm',
    'request',
    'Pillow',
]

setup(
    # Metadata
    name='gluoncv-torch',
    version=version,
    author='Gluon CV Toolkit Contributors',
    url='https://github.com/dmlc/gluon-cv',
    description='MXNet Gluon CV Toolkit',
    long_description=long_description,
    license='Apache-2.0',

    # Package info
    packages=find_packages(exclude=('docs', 'tests', 'scripts')),

    zip_safe=True,
    include_package_data=True,
    install_requires=requirements,
)
