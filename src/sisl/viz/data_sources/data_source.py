# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from nodify import Node


class DataSource(Node):
    """Generic class for data sources.

    Data sources are a way of specifying and manipulating data without providing it explicitly.
    Data sources can be passed to the settings of the plots as if they were arrays.
    When the plot is being created, the data source receives the necessary inputs and is evaluated using
    its ``get`` method.

    Therefore, passing a data source is like passing a function that will receive
    inputs and calculate the values needed on the fly. However, it has some extra functionality. You can
    perform operations with a data source. These operations will be evaluated lazily, that is, when
    inputs are provided. That allows for very convenient manipulation of the data.

    Data sources are also useful for graphical interfaces, where the user is unable to explicitly
    pass a function. Some of them are
    """

    pass
