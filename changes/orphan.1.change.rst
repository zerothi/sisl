Removed `dtype` argument from `Spin` class

The data-type is now contained in the parent structure.
This removes a duplicate definition that was hard to maintain
in the code. It should be of minor importance as most would
define the `Spin` class without passing the `dtype` argument.
