#!/usr/bin/env python

from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here/'README.md').read_text(encoding='utf-8')

setup(
      name='optimization',
      version='0.1.2',
      description='Optimization algorithms',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Mário antunes',
      author_email='mariolpantunes@gmail.com',
      url='https://github.com/mariolpantunes/optimization',
      packages=find_packages(),
      install_requires=['numpy>=1.22.1', 'joblib>=1.1.0', 'tqdm>=4.62.3']
)
