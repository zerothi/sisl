# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from abc import abstractmethod

from sisl._internal import set_module

__all__ = ["Metric", "CompositeMetric"]


@set_module("sisl_toolbox.siesta.minimizer")
class Metric:
    @abstractmethod
    def metric(self, variables, *args, **kwargs):
        """Return a single number quantifying the metric of the system"""

    def __abs__(self):
        return AbsMetric(self)

    def __add__(self, other):
        return SumMetric(self, other)

    def __sub__(self, other):
        return SubMetric(self, other)

    def __rsub__(self, other):
        return SubMetric(other, self)

    def __mul__(self, factor):
        return MulMetric(self, factor)

    def __truediv__(self, divisor):
        return DivMetric(self, divisor)

    def __rtruediv__(self, divisor):
        return DivMetric(divisor, self)

    def __neg__(self):
        return MulMetric(-1, self)

    def __pow__(self, other, mod=None):
        return PowMetric(self, other, mod)

    def __rpow__(self, other, mod=None):
        return PowMetric(other, self, mod)

    def min(self, other):
        return MinMetric(self, other)

    def max(self, other):
        return MaxMetric(self, other)


@set_module("sisl_toolbox.siesta.minimizer")
class CompositeMetric(Metric):
    """Placeholder for two metrics"""

    def __init__(self, A, B):
        self.A = A
        self.B = B

    def _metric_composite(self, variables, *args, **kwargs):
        if isinstance(self.A, Metric):
            A = self.A.metric(variables, *args, **kwargs)
        else:
            A = self.A
        if isinstance(self.B, Metric):
            B = self.B.metric(variables, *args, **kwargs)
        else:
            B = self.B
        return A, B


@set_module("sisl_toolbox.siesta.minimizer")
class AbsMetric(Metric):
    def __init__(self, A):
        self.A = A

    def metric(self, variables, *args, **kwargs):
        return abs(super().metric(variables, *args, **kwargs))


@set_module("sisl_toolbox.siesta.minimizer")
class MinMetric(CompositeMetric):
    def metric(self, variables, *args, **kwargs):
        A, B = self._metric_composite(variables, *args, **kwargs)
        return min(A, B)


@set_module("sisl_toolbox.siesta.minimizer")
class MaxMetric(CompositeMetric):
    def metric(self, variables, *args, **kwargs):
        A, B = self._metric_composite(variables, *args, **kwargs)
        return max(A, B)


@set_module("sisl_toolbox.siesta.minimizer")
class SumMetric(CompositeMetric):
    def metric(self, variables, *args, **kwargs):
        A, B = self._metric_composite(variables, *args, **kwargs)
        return A + B


@set_module("sisl_toolbox.siesta.minimizer")
class SubMetric(CompositeMetric):
    def metric(self, variables, *args, **kwargs):
        A, B = self._metric_composite(variables, *args, **kwargs)
        return A - B


@set_module("sisl_toolbox.siesta.minimizer")
class MulMetric(CompositeMetric):
    def metric(self, variables, *args, **kwargs):
        A, B = self._metric_composite(variables, *args, **kwargs)
        return A * B


@set_module("sisl_toolbox.siesta.minimizer")
class DivMetric(CompositeMetric):
    def metric(self, variables, *args, **kwargs):
        A, B = self._metric_composite(variables, *args, **kwargs)
        return A / B


@set_module("sisl_toolbox.siesta.minimizer")
class PowMetric(CompositeMetric):
    def __init__(self, A, B, mod=None):
        super().__init__(A, B)
        self.mod = mod

    def metric(self, variables, *args, **kwargs):
        A, B = self._metric_composite(variables, *args, **kwargs)
        if self.mod is None:
            return pow(A, B)
        return pow(A, B, self.mod)
