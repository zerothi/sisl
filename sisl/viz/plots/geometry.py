from functools import wraps
import numpy as np

from sisl import Geometry
from sisl.viz import Plot
from sisl.viz.input_fields import ProgramaticInput, FloatInput, SwitchInput
from sisl.viz.GUI.api_utils._dispatcher import AbstractDispatch, ClassDispatcher

class BoundGeometry(AbstractDispatch):
    '''
    Updates the plot after a method is run on the plot's geometry.
    '''

    def __init__(self, geom, parent_plot):

        self.parent_plot = parent_plot
        super().__init__(geom)

    def dispatch(self, method):

        @wraps(method)
        def with_plot_update(*args, **kwargs):

            ret = method(*args, **kwargs)

            # Maybe the returned value is not a geometry
            if isinstance(ret, Geometry):
                self.parent_plot.update_settings(geom=ret)
                return self.parent_plot.on_geom

            return ret

        return with_plot_update

class BaseGeometryPlot(Plot):
    '''
    Representation of a geometry in a plotly Figure.
    
    This class serves just as a base for child classes that display properties of a geometry.
    IF YOU WANT TO BUILD A GEOMETRY BASED PLOT, INHERIT FROM THIS CLASS. It contains all the necessary
    methods for this purpose.

    However, this class IS NOT TO BE USED DIRECTLY. In this way, we can keep child classes
    clean of meaningless settings. TO DISPLAY A GEOMETRY, USE GEOMETRY PLOT.

    Warning: by now, make sure _after_read is triggered (i.e. if you overwrite it
    in your class, call this classes' one explicitly)
    '''

    is_only_base = True
    
    _parameters = (
        ProgramaticInput(key="geom", name="Geometry",
            default=None
        ),

        SwitchInput(key='bonds', name='Show bonds',
            default=True,
        ),

        SwitchInput(key='cell', name='Show cell',
            default=True,
        ),
    )
    
    _overwrite_defaults = {
        'xaxis_showgrid': False,
        'yaxis_showgrid': False
    }

    @property
    def on_geom(self):
        return BoundGeometry(self.geom, self)

    def _after_init(self):

        self.bonds = None
    
    @staticmethod
    def _sphere(center=[0,0,0], r=1):
        phi, theta = np.mgrid[0.0:np.pi:10j, 0.0:2.0*np.pi:10j]
        x = center[0] + r*np.sin(phi)*np.cos(theta)
        y = center[1] + r*np.sin(phi)*np.sin(theta)
        z = center[2] + r*np.cos(phi)
    
        return {'x': x,'y': y,'z': z}
    
    def _read_nosource(self):
        self.geom = self.setting("geom") or getattr(self, "geom", None)
        
        if self.geom is None:
            raise Exception("No geometry has been provided.")       
            
    def _after_read(self):

        if self.setting('bonds'):
            self.bonds = self.find_all_bonds(self.geom)
    
    @staticmethod
    def find_all_bonds(geom):
        
        bonds = []
        for at in geom:
            neighs = geom.close(at, R=[0.1, 2])[-1]

            for neigh in neighs:
                bonds.append(np.sort([at, neigh]))

        if bonds:
            return np.unique(bonds, axis=0)
        else:
            return bonds

    def _default_wrap_atom3D(self, at):

        return (self.geom[at], 0.3), {"name":f'{at+1} ({self.geom.atom[at].tag})'}

    def _default_wrap_bond3D(self, bond):

        return (*self.geom[bond], 15), {}

    def _plot_geom3D(self, wrap_atom=None, wrap_bond=None):
        '''
        Returns a 3D representation of the plot's geometry.

        Parameters
        -----------
        wrap_atom: function, optional
            function that recieves the index of an atom and returns
            the args (array-like) and kwargs (dict) that go into self._atom_trace3D()

            If not provided, self._default_wrap_atom3D will be used.
        wrap_bond: function, optional
            function that recieves "a bond" (list of 2 atom indices) and returns
            the args (array-like) and kwargs (dict) that go into self._bond_trace3D()

            If not provided, self._default_wrap_bond3D will be used.
        '''

        wrap_atom = wrap_atom or self._default_wrap_atom3D
        wrap_bond = wrap_bond or self._default_wrap_bond3D
        
        # Add atoms
        atom_traces = []
        for at in self.geom:
            trace_args, trace_kwargs = wrap_atom(at)
            atom_traces.append(self._atom_trace3D(*trace_args, **trace_kwargs))
        self.add_traces(atom_traces)
        
        # Add bonds
        if self.setting('bonds'):
            bond_traces = []
            for bond in self.bonds:
                trace_args, trace_kwargs = wrap_bond(bond)
                bond_traces.append(self._bond_trace3D(*trace_args, **trace_kwargs))
            self.add_traces(bond_traces)

        # Draw unit cell
        if self.setting('cell'):
            self.add_traces(self._cell_axes_traces3D())
            
        forbidden3D_keys = ('title', 'showlegend', 'paper_bgcolor', 'plot_bgcolor',
                          'xaxis_scaleanchor', 'xaxis_scaleratio', 'yaxis_scaleanchor', 'yaxis_scaleratio')
        
        self.layout.scene = {
            'aspectmode': 'data', 
            **{key:val for key, val in self.settings_group("layout").items() if key not in forbidden3D_keys}
        }

    def _cell_axes_traces3D(self, cell=None):

        return [{
            'type': 'scatter3d',
            'x': [0, vec[0]],
            'y': [0, vec[1]],
            'z': [0, vec[2]],
            'name': f'Axis {i}'
        } for i, vec in enumerate(cell or self.geom.cell)]
    
    def _cell_trace3D(self, cell):
        pass
        
    def _atom_trace3D(self, xyz, r, color="gray", name=None, group=None, showlegend=False, **kwargs):
        
        trace = {
            'type': 'surface',
            **self._sphere(xyz, r),
            'showlegend': showlegend,
            'colorscale': [[0, color], [1, color]],
            'showscale': False,
            'legendgroup': group,
            'name': name,
            'meta': ['({:.2f}, {:.2f}, {:.2f})'.format(*xyz)],
            'hovertemplate': '%{meta[0]}',
            **kwargs
        }
        
        return trace
    
    def _bond_trace2D(self, xyz1, xyz2, r=15, color="gray", name=None, group=None, showlegend=False, **kwargs):
        
        return trace
        
    def _bond_trace3D(self, xyz1, xyz2, r=15, color="#ccc", name=None, group=None, showlegend=False, **kwargs):
        
        # Drawing cylinders instead of bonds will be better, but rendering would be slower
        # We need to give the possibility.

        x, y, z = np.array([xyz1, xyz2]).T
            
        trace = {
            'type': 'scatter3d',
            'mode': 'lines',
            'name': name,
            'x': x,
            'y': y,
            'z': z,
            'line': {'width': r, 'color': color},
            'legendgroup': group,
            'showlegend': showlegend,
            **kwargs
        }
        
        return trace

class GeometryPlot(BaseGeometryPlot):

    _plot_type = "Geometry"

    def _set_data(self):

        self._plot_geom3D()



