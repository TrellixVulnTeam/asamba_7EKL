
# Asteroseismic Modelling Database Python Tools

[TOC]

## About

## Cloning the Repository
To clone the latest public release of the repository, you may simply do the following:

```bash
cd <path-to-save-repository>
git clone https://ehsan_moravveji@bitbucket.org/ehsan_moravveji/grid.git
```

## Documentation
All modules and functions in the repository are documented. Thus, it is easy to read the intent of each module/function/class objects, by referring to the source code. An alternative way to read the basic documentation of the class objects is to instantiate them, and then accessing the `__doc__` attribute. As an example, to read the documentation for the `model` object, one can do the following:

```python
from grid import var_def
a_model = var_def.model()
print a_model.__doc__
```
Soon, an HTML documentation page will be also added to the repository for a more convenient representation of the entire documented modules, functions and classes.

## References