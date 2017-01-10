
# Asteroseismic Modelling Database Python Tools

[TOC]

## About
The `grid` is a Python interface to interact with the forward asteroseismic grid of massive stars, computed by the MESA stellar structure and evolution code, coupled with the GYRE adiabatic nonradial pulsation code. The whole data in the grid is organized as a PostgreSQL database, and the `grid` module allows the users to exploit this database.

The computation of the grid, the development of the database, and the development of this repository are currently under intense development. So, things will grow/improve steeply soon.

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

## Notes About The Data Structure
Below, I provide few key notes about how the data is organized in this package.

* The attribute names of the `model` classe is borrowed from their exact names in the MESA output history file. These names also match the attributes of the database table `models`. Therefore, MESA users have easy time understanding the meaning of the data in the database/repository.

* The physical units of all variables follow closely their units in MESA. E.g. the initial mass is given in solar unit, while the temperature (and other quantities) is given in CGS units. Thus, a proper reference to figure out the unit of each parameter is by consulting the `<mesa>/star/defaults/history_columns.list` file.

## References