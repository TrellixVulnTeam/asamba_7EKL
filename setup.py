#!/usr/bin/env python

"""
This script handles the installation of the module across any platform which are 
compatible with both Python 2.7 and Python 3.6.
"""

from __future__ import unicode_literals

from setuptools import setup
import glob

setup(name='asamba',
      version='1.0.5',
      author='Ehsan Moravveji',
      description='Asteroseismic Modelling Database Python Tools',
      keywords='Asteroseismology, Pulsating Massive Stars, Modelling',
      author_email='Ehsan.Moravveji@kuleuven.be',
      # url='git@github.com:moravveji/asamba.git',
      url='https://fys.kuleuven.be/ster/Projects/ASAMBA',
      license='GPL',
      # package_dir={'':'asamba'}, 
      packages=['asamba'],
      py_modules=['star', 'utils', 'read', 'write', 
                  'var_def', 'var_lib', 'db_def', 'db_lib', 
                  'query', 'insert_def', 'insert_lib',
                  'sampler', 'artificial_neural_network', 'interpolator', 
                  'plot_sampler', 'plot_interpolator',
                  'backend', 'frontend'],
      classifiers=['Development Status :: 4 - Beta',
                   'Environment :: MacOS X',
                   'Environment :: Win32 (MS Windows)',
                   'Environment :: Other Environment',
                   'Framework :: IPython',
                   'Intended Audience :: Science/Research',
                   'Intended Audience :: Education',
                   'Intended Audience :: End Users/Desktop',
                   'License :: Free For Educational Use',
                   'License :: Free For Home Use',
                   'License :: Free for non-commercial use',
                   'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
                   'Natural Language :: English',
                   'Operating System :: MacOS :: MacOS X',
                   'Operating System :: POSIX :: Linux',
                   'Operating System :: Microsoft :: Windows :: Windows 7',
                   # 'Operating System :: Microsoft :: Windows :: Windows 10',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3.6',
                   'Topic :: Scientific/Engineering :: Astronomy',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence',
                   'Topic :: Software Development :: Version Control :: Git',
                   ],
      install_requires=['h5py', 'psycopg2', 'numpy', 'scipy'],
     )

