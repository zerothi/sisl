from .geometry import GeometryBackend

from ....plots import BondLengthMap

class BondLengthMapBackend(GeometryBackend):

    def draw_1D(self, backend_info, **kwargs):
        return NotImplementedError("1D representations of bond length maps are not implemented")

BondLengthMap._backends.register_template(BondLengthMapBackend)