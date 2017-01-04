
exec(open('version.py').read())
setup(
    ...
    version=__version__,
    ...)