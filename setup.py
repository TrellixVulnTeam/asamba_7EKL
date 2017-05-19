#!/usr/bin/env python

"""
This script handles the installation of the module across any platform which are 
compatible with both Python 2.7 and Python 3.6.
"""

from __future__ import unicode_literals

from setuptools import setup
import glob

description = '"A"stero"S"eismic "A"pproach in "M"odelling "B"lue st"A"rs (ASAMBA) is a Marie Curie project \
              that tries to infer deep physical understanding of the internal structure and evolution of massive \
              stars in the light of recent very high precision space observations of pulsating massive stars.\n \
              Under this umbrella, a large grid of stellar models (using MESA) are computed, and the theoretical \
              pulsation frequencies of each model (after iterating over various rotation rates) are also computed\
              using the (GYRE) code. From this rich dataset (~3.8 million stellar models, and 42 million frequency\
              lists), a PostgreSQL database is built, and is made openly accessible.\n \
              The present Python package provides a convenient user interface that offers all available functionalities \
              to the users, and allows them to interact with the database, and conduct their own research of interest. \
              Needless to say that the user must have a full understanding of the meaning of the parameters he/she uses \
              which steer the analysis. We strongly recommend reading the source code, the documentation around most of \
              the code blocks, and the compiled documentation pages that ships in with this package.'

setup(name='asamba',
      version='1.0.7',
      author='Ehsan Moravveji',
      description=description,
      keywords='Asteroseismology, Pulsating Massive Stars, Modelling',
      author_email='Ehsan.Moravveji@kuleuven.be',
      # url='git@github.com:moravveji/asamba.git',
      url='https://fys.kuleuven.be/ster/Projects/ASAMBA',
      license='GPL',
      packages=['asamba', 'test_suite'],
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
                   'Operating System :: Microsoft :: Windows :: Windows 10',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3.6',
                   'Topic :: Scientific/Engineering :: Astronomy',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence',
                   'Topic :: Software Development :: Version Control :: Git',
                   ],
      install_requires=['h5py', 'psycopg2', 'numpy', 'scipy'],
     )

