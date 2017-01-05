
#!/usr/bin/env python

from distutils.core import setup

exec(open('version.py').read())

setup(name='grid',
      version=__version__,
      description='Asteroseismic Modelling Database Python Tools',
      author='Ehsan Moravveji',
      author_email='',
      url='https://ehsan_moravveji@bitbucket.org/ehsan_moravveji/grid.git',
      # packages=['distutils'],
     )

