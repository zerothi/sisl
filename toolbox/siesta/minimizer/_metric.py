from abc import abstractmethod


__all__ = ["Metric", "CompositeMetric"]


class Metric:

    @abstractmethod
    def metric(self, variables, *args, **kwargs):
        """ Return a single number quantifying the metric of the system """

    def __add__(self, other):
        return SumMetric(self, other)

    def __sub__(self, other):
        return SubMetric(self, other)

    def __mul__(self, factor):
        return MulMetric(self, factor)

    def __truediv__(self, divisor):
        return MulMetric(self, 1. / divisor)

    def __neg__(self):
        return MulMetric(-1, self)

    def min(self, other):
        return MinMetric(self, other)

    def max(self, other):
        return MaxMetric(self, other)


class CompositeMetric(Metric):
    """ Placeholder for two metrics """

    def __init__(self, A, B):
        self.A = A
        self.B = B

    def _metric_composite(self, variables, *args, **kwargs):
        if isinstance(self.A, Metric):
            A = self.A(variables, *args, **kwargs)
        else:
            A = self.A
        if isinstance(self.B, Metric):
            B = self.B(variables, *args, **kwargs)
        else:
            B = self.B
        return A, B


class MinMetric(CompositeMetric):
    def metric(self, variables, *args, **kwargs):
        A, B = self._metric_composite(variables, *args, **kwargs)
        return min(A, B)


class MaxMetric(CompositeMetric):
    def metric(self, variables, *args, **kwargs):
        A, B = self._metric_composite(variables, *args, **kwargs)
        return max(A, B)


class SumMetric(CompositeMetric):
    def metric(self, variables, *args, **kwargs):
        A, B = self._metric_composite(variables, *args, **kwargs)
        return A + B


class SubMetric(CompositeMetric):
    def metric(self, variables, *args, **kwargs):
        A, B = self._metric_composite(variables, *args, **kwargs)
        return A - B


class MulMetric(CompositeMetric):
    def metric(self, variables, *args, **kwargs):
        A, B = self._metric_composite(variables, *args, **kwargs)
        return A * B
