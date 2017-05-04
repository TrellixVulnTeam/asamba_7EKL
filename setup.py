#!/usr/bin/env python

"""
This script handles the installation of the module across any platform which are 
compatible with both Python 2.7 and Python 3.6.
"""

from __future__ import unicode_literals

from distutils.core import setup
import glob

setup(name='grid',
      version='1.01',
      author='Ehsan Moravveji',
      description='Asteroseismic Modelling Database Python Tools',
      keywords='asteroseismology massive stars modelling',
      author_email='Ehsan.Moravveji@kuleuven.be',
      url='https://ehsan_moravveji@bitbucket.org/ehsan_moravveji/grid.git',
      license='GPL',
      # package_dir={'':''}, 
      # packages=['distutils'],
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
                   'Operating System :: Microsoft :: Windows :: Windows 10',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3.6',
                   'Topic :: Scientific/Engineering :: Astronomy',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence',
                   'Topic :: Software Development :: Version Control :: Git'
                   ]
     )

