import sisl
from sisl.viz import BondLengthMap
from sisl.viz.plots.tests.test_geometry import 

# ------------------------------------------------------------
#      Build a generic tester for bond length plot
# ------------------------------------------------------------

class ForcesPlotTester(GeometryPlotTester):
    pass

# ------------------------------------------------------------
#            Test it with two sisl geometries
# ------------------------------------------------------------


class TestSislBondLength(BondLengthMapTester):

    plot = BondLengthMap(
        geom=sisl.geom.graphene(orthogonal=True, bond=1.35),
        strain_ref=sisl.geom.graphene(orthogonal=True)
    )
    has_strain_ref = True
