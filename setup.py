#coding: utf8

"""
Setup script for speckles simulations.
"""

from glob import glob

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup(name='speckles_simulations',
      version='0.0.1',
      author="fperakis",
      author_email="fperakis@fysik.su.se",
      description='XPCS code',
      packages=["speckles-simulations", "speckle-simulations"],
      package_dir={"speckle-simulations": "speckle-simulations"},
      scripts=[s for s in glob('scripts/*') if not s.endswith('__.py')],
test_suite="test")
