# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from sisl._core._dtypes cimport floats_st


cdef bint is_gamma(const floats_st[::1] k) noexcept nogil
