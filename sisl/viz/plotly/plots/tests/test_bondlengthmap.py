import sisl
from sisl.viz import BondLengthMap
from sisl.viz.plotly.plots.tests.test_geometry import GeometryPlotTester

# ------------------------------------------------------------
#      Build a generic tester for bond length plot
# ------------------------------------------------------------


class BondLengthMapTester(GeometryPlotTester):

    has_strain_ref = False

    def test_strain_ref(self):

        plot = self.plot

        if self.has_strain_ref:
            plot.update_settings(axes=[0, 1, 2], strain=True, bonds=True)

            strains = plot.data[0].line.color

            plot.update_settings(strain=False)
            assert plot.data[0].line.color != strains

# ------------------------------------------------------------
#            Test it with two sisl geometries
# ------------------------------------------------------------


class TestSislBondLength(BondLengthMapTester):

    plot = BondLengthMap(
        geometry=sisl.geom.graphene(orthogonal=True, bond=1.35),
        strain_ref=sisl.geom.graphene(orthogonal=True)
    )
    has_strain_ref = True
