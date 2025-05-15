# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from typing import Union

import numpy as np

from sisl._internal import set_module
from sisl.typing import SpinType

__all__ = ["Spin"]


@set_module("sisl.physics")
class Spin:
    r"""Spin class to determine configurations and spin components.

    The basic class `Spin` implements a generic method to determine a spin configuration.

    Its usage can be summarized in these few examples:

    >>> Spin(Spin.UNPOLARIZED) == Spin("unpolarized") == Spin()
    True
    >>> Spin(Spin.POLARIZED) == Spin("polarized") == Spin("p")
    True
    >>> Spin(Spin.NONCOLINEAR, dtype=np.complex128) == Spin("non-collinear") == Spin("nc")
    True
    >>> Spin(Spin.SPINORBIT, dtype=np.complex128) == Spin("spin-orbit") == Spin("so") == Spin("soc")
    True
    >>> Spin(Spin.NAMBU) == Spin("nambu") == Spin("bdg")
    True

    Note that a data-type may be associated with a spin-object. This is not to say
    that the data-type is used in the configuration, but merely that it helps
    any sub-classed or classes who use the spin-object to determine the
    usage of the different spin-components.

    Parameters
    ----------
    kind : str or int, Spin, optional
       specify the spin kind
    """

    #: Constant for an un-polarized spin configuration
    UNPOLARIZED = 0
    """Constant for an un-polarized spin configuration."""
    #: Constant for a polarized spin configuration
    POLARIZED = 1
    """Constant for a polarized spin configuration."""
    #: Constant for a non-collinear spin configuration
    NONCOLINEAR = 2
    """Constant for a non-collinear spin configuration."""
    #: Constant for a spin-orbit spin configuration
    SPINORBIT = 3
    """Constant for a spin-orbit spin configuration."""
    #: Constant for a Nambu spin configuration
    NAMBU = 4
    """Constant for a Nambu spin configuration."""

    #: The :math:`\boldsymbol\sigma_x` Pauli matrix
    X = np.array([[0, 1], [1, 0]], np.complex128)
    #: The :math:`\boldsymbol\sigma_y` Pauli matrix
    Y = np.array([[0, -1j], [1j, 0]], np.complex128)
    #: The :math:`\boldsymbol\sigma_z` Pauli matrix
    Z = np.array([[1, 0], [0, -1]], np.complex128)

    __slots__ = ("_kind",)

    def __init__(self, kind: SpinType = "unpolarized"):
        if isinstance(kind, Spin):
            self._kind = kind._kind
            return

        if isinstance(kind, str):
            kind = kind.lower()

        kind = {
            "": Spin.UNPOLARIZED,
            "unpolarized": Spin.UNPOLARIZED,
            Spin.UNPOLARIZED: Spin.UNPOLARIZED,
            "colinear": Spin.POLARIZED,
            "collinear": Spin.POLARIZED,
            "polarized": Spin.POLARIZED,
            "p": Spin.POLARIZED,
            "pol": Spin.POLARIZED,
            Spin.POLARIZED: Spin.POLARIZED,
            "noncolinear": Spin.NONCOLINEAR,
            "noncollinear": Spin.NONCOLINEAR,
            "non-colinear": Spin.NONCOLINEAR,
            "non-collinear": Spin.NONCOLINEAR,
            "nc": Spin.NONCOLINEAR,
            Spin.NONCOLINEAR: Spin.NONCOLINEAR,
            "spinorbit": Spin.SPINORBIT,
            "spin-orbit": Spin.SPINORBIT,
            "so": Spin.SPINORBIT,
            "soc": Spin.SPINORBIT,
            Spin.SPINORBIT: Spin.SPINORBIT,
            "nambu": Spin.NAMBU,
            "bdg": Spin.NAMBU,
            Spin.NAMBU: Spin.NAMBU,
        }.get(kind)
        if kind is None:
            raise ValueError(
                f"{self.__class__.__name__} initialization went wrong because of wrong "
                "kind specification. Could not determine the kind of spin!"
            )

        # Now assert the checks
        self._kind = kind

    def __str__(self) -> str:
        if self.is_unpolarized:
            return f"{self.__class__.__name__}{{unpolarized}}"
        if self.is_polarized:
            return f"{self.__class__.__name__}{{polarized}}"
        if self.is_noncolinear:
            return f"{self.__class__.__name__}{{non-colinear}}"
        if self.is_spinorbit:
            return f"{self.__class__.__name__}{{spin-orbit}}"
        return f"{self.__class__.__name__}{{nambu}}"

    def __repr__(self) -> str:
        if self.is_unpolarized:
            return f"<{self.__class__.__name__} unpolarized>"
        if self.is_polarized:
            return f"<{self.__class__.__name__} polarized>"
        if self.is_noncolinear:
            return f"<{self.__class__.__name__} non-colinear>"
        if self.is_spinorbit:
            return f"<{self.__class__.__name__} spin-orbit>"
        return f"<{self.__class__.__name__} nambu>"

    def copy(self):
        """Create a copy of the spin-object"""
        return Spin(self.kind)

    def size(self, dtype: np.dtype) -> int:
        """Number of elements to describe the spin-components

        Parameters
        ----------
        dtype:
            data-type used to represent the spin-configuration
        """
        dkind = np.dtype(dtype).kind
        if dkind == "c":
            return {
                self.UNPOLARIZED: 1,
                self.POLARIZED: 2,
                self.NONCOLINEAR: 3,
                self.SPINORBIT: 4,
                self.NAMBU: 8,
            }[self.kind]

        return {
            self.UNPOLARIZED: 1,
            self.POLARIZED: 2,
            self.NONCOLINEAR: 4,
            self.SPINORBIT: 8,
            self.NAMBU: 16,
        }[self.kind]

    @property
    def spinor(self) -> int:
        """Number of spinor components (1, 2 or 4)"""
        if self.is_unpolarized:
            return 1
        if self.is_nambu:
            return 4
        return 2

    @property
    def kind(self) -> int:
        """A unique ID for the kind of spin configuration"""
        return self._kind

    @property
    def is_unpolarized(self) -> bool:
        """True if the configuration is not polarized"""
        # Regardless of data-type
        return self.kind == Spin.UNPOLARIZED

    @property
    def is_polarized(self) -> bool:
        """True if the configuration is polarized"""
        return self.kind == Spin.POLARIZED

    is_colinear = is_polarized

    @property
    def is_noncolinear(self) -> bool:
        """True if the configuration non-collinear"""
        return self.kind == Spin.NONCOLINEAR

    @property
    def is_diagonal(self) -> bool:
        """Whether the spin-box is only using the diagonal components

        This will return true for non-polarized and polarized spin configurations.
        Otherwise false.
        """
        return self.kind in (Spin.UNPOLARIZED, Spin.POLARIZED)

    @property
    def is_spinorbit(self) -> bool:
        """True if the configuration is spin-orbit"""
        return self.kind == Spin.SPINORBIT

    @property
    def is_nambu(self) -> bool:
        """True if the configuration is Nambu"""
        return self.kind == Spin.NAMBU

    # Comparisons
    def __lt__(self, other) -> bool:
        return self.kind < other.kind

    def __le__(self, other) -> bool:
        return self.kind <= other.kind

    def __eq__(self, other) -> bool:
        return self.kind == other.kind

    def __ne__(self, other) -> bool:
        return not self == other

    def __gt__(self, other) -> bool:
        return self.kind > other.kind

    def __ge__(self, other) -> bool:
        return self.kind >= other.kind

    def __getstate__(self) -> dict:
        return {"kind": self.kind}

    def __setstate__(self, state):
        self._kind = state["kind"]
