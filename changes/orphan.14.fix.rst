`BrillouinZone.apply.array` accepts scalar values

``bz.apply.array`` now accepts methods (or ``wrap`` functions) that return a Python
scalar. Before, it raised an `AttributeError`.
