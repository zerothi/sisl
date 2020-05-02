from functools import wraps
import numpy as np

from sisl import Geometry, PeriodicTable
from sisl.viz import Plot
from sisl.viz.input_fields import ProgramaticInput, FloatInput, SwitchInput, DropdownInput, TextInput
from sisl._dispatcher import AbstractDispatch, ClassDispatcher

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

        TextInput(key="geom_file", name="Geometry file",
            default=None
        ),

        SwitchInput(key='bonds', name='Show bonds',
            default=True,
        ),
    )
    
    _overwrite_defaults = {
        'xaxis_showgrid': False,
        'xaxis_zeroline': False,
        'yaxis_showgrid': False,
        'yaxis_zeroline': False,
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

    def _read_siesta_output(self):

        geom_file = self.setting("geom_file")

        self.geom = self.get_sile(geom_file).read_geometry()
            
    def _after_read(self):

        if self.setting('bonds'):
            self.bonds = self.find_all_bonds(self.geom)
    
    @staticmethod
    def find_all_bonds(geom):
        
        pt = PeriodicTable()

        bonds = []
        for at in geom:
            neighs = geom.close(at, R=[0.1, 3])[-1]

            for neigh in neighs:
                if pt.radius([geom.atom[at].Z, geom.atom[neigh % geom.na].Z]).sum() + 0.15 > np.linalg.norm(geom[neigh] - geom[at]):
                    bonds.append(np.sort([at, neigh]))

        if bonds:
            return np.unique(bonds, axis=0)
        else:
            return bonds

    def _plot_geom(self, atom_trace, bond_trace, cell_axes_traces=None, cell_trace=None, wrap_atom=None, wrap_bond=None, atom=None, cell="axes"):

        # Add atoms
        atom_traces = []
        ats = self.geom[atom] if atom is not None else self.geom
        for at in self.geom:
            trace_args, trace_kwargs = wrap_atom(at)
            atom_traces.append(atom_trace(*trace_args, **trace_kwargs))
            self.add_traces(atom_traces)

        # Add bonds
        if self.setting('bonds'):
            bond_traces = []

            bonds = self.bonds
            if atom is not None:
                bonds = [bond for bond in bonds if np.any([at in ats for at in bond])]  

            for bond in self.bonds:
                trace_args, trace_kwargs = wrap_bond(bond)
                bond_traces.append(bond_trace(*trace_args, **trace_kwargs))
            self.add_traces(bond_traces)

        # Draw unit cell
        if cell == "axes":
            self.add_traces(cell_axes_traces())
        elif cell == "box":
            self.add_trace(cell_trace())
    
    def _sanitize_axis(self, axis):

        if isinstance(axis, str):
            try:
                i = ["x", "y", "z"].index(axis)
                axis = np.zeros(3)
                axis[i] = 1
            except:
                i = ["a", "b", "c"].index(axis)
                axis = self.geom.cell[i]
        elif isinstance(axis, int):
            axis = self.geom.cell[axis]
        
        return np.array(axis)
    
    def _get_cell_corners(self, cell=None, unique=False):

        if cell is None:
            cell = self.geom.cell

        def xyz(coeffs):
            return np.dot(coeffs, cell)

        # Define the vertices of the cube
        points = [
            (0,0,0), (0,1,0), (1,1,0), (1,0,0), (0,0,0),
            (0,0,1), (0,1,1), (0,1,0), (0,1,1), (1,1,1),
            (1,1,0), (1,0,0), (1,0,1), (1,1,1), (1,0,1), (0,0,1)
        ]

        if unique:
            points = np.unique(points, axis=0)

        return np.array([xyz(coeffs) for coeffs in points])

    #---------------------------------------------------
    #                  2D plotting
    #---------------------------------------------------

    def _plot_geom2D(self, xaxis="x", yaxis="y", bonds=True, cell='axes'):
        '''
        Returns a 2D representation of the plot's geometry.

        Parameters
        -----------
        ort_vec: array-like of length 3
            A vector representing the direction orthogonal to the plot canvas.
            The point of view, so to speak.
        wrap_atom: function, optional
            function that recieves the index of an atom and returns
            the args (array-like) and kwargs (dict) that go into self._atom_trace3D()

            If not provided, self._default_wrap_atom2D will be used.
        wrap_bond: function, optional
            function that recieves "a bond" (list of 2 atom indices) and returns
            the args (array-like) and kwargs (dict) that go into self._bond_trace3D()

            If not provided, self._default_wrap_bond2D will be used.
        '''

        xy = self._projected_2Dcoords(self.geom.xyz, xaxis=xaxis, yaxis=yaxis)
        traces = []

        # Add bonds
        if bonds:
            for bond in self.bonds:
                xys = self._projected_2Dcoords(self.geom[bond], xaxis=xaxis, yaxis=yaxis)
                trace = self._bond_trace2D(*xys.T)
                traces.append(trace)

        traces.append(
            self._atoms_scatter_trace2D(
                xy, 
                text=[f'{self.geom[at]}<br>{at+1} ({self.geom.atom[at].tag})' for at in self.geom], name="Atoms"
            )
        )

        if cell == "box":
            traces.append(self._cell_trace2D(xaxis=xaxis, yaxis=yaxis))
        if cell == "axes":
            traces = [*traces, *self._cell_axes_traces2D(xaxis=xaxis, yaxis=yaxis)]

        self.add_traces(traces)

    def _projected_2Dcoords(self, xyz=None, xaxis="x" , yaxis="y"):
        '''
        Moves the 3D positions of the atoms to a 2D supspace.

        In this way, we can plot the structure from the "point of view" that we want.

        Parameters
        ------------
        xyz: array-like of shape (natoms, 3), optional
            the 3D coordinates that we want to project.
            otherwise 
        xaxis: {0,1,2, "x", "y", "z", "a", "b", "c"} or array-like of shape 3, optional
            the direction to be displayed along the X axis. 
            If it's an int, it will interpreted as the index of the cell axis.
        yaxis: {0,1,2, "x", "y", "z", "a", "b", "c"} or array-like of shape 3, optional
            the direction to be displayed along the X axis. 
            If it's an int, it will interpreted as the index of the cell axis.

        Returns
        ----------
        np.ndarray of shape (2, natoms)
            the 2D coordinates of the geometry, with all positions projected into the plane
            defined by xaxis and yaxis.
        '''
        if xyz is None:
            xyz = self.geom.xyz

        # Get the directions that these axes represent if the provided input
        # is an axis index
        xaxis = self._sanitize_axis(xaxis)
        yaxis = self._sanitize_axis(yaxis)

        return np.array([xyz.dot(ax)/np.linalg.norm(ax) for ax in (xaxis, yaxis)])

    def _atom_circle_trace2D(self, xyz, r):

        raise NotImplementedError

    def _atoms_scatter_trace2D(self, xyz=None, xaxis="x", yaxis="y", color="gray", size=10, name=None, text=None, group=None, showlegend=False, **kwargs):

        if xyz is None:
            xyz = self.geom.xyz

        # If 3D coordinates 3D coordinates into 2D
        if xyz.shape[1] == 3:
            xy = self._projected_2Dcoords(xyz)
        else:
            xy = xyz
        
        trace = {
            "type": "scatter",
            'mode': 'markers',
            'name': name,
            'x': xy[0],
            'y': xy[1],
            'marker': {'size': size, 'color': color},
            'text': text,
            'legendgroup': group,
            'showlegend': showlegend,
            **kwargs
        }

        return trace

    def _bond_trace2D(self, xy1, xy2, width=2, color="#ccc", name=None, group=None, showlegend=False, **kwargs):
        '''
        Returns a bond trace in 2d.
        '''

        x, y = np.array([xy1, xy2]).T

        trace = {
            "type": "scatter",
            'mode': 'lines',
            'name': name,
            'x': x,
            'y': y,
            'line': {'width': width, 'color': color},
            'legendgroup': group,
            'showlegend': showlegend,
            **kwargs
        }

        return trace

    def _cell_axes_traces2D(self, cell=None, xaxis="x", yaxis="y"):

        if cell is None:
            cell = self.geom.cell

        cell_xy = self._projected_2Dcoords(xyz=cell, xaxis=xaxis, yaxis=yaxis).T

        return [{
            'type': 'scatter',
            'mode': 'markers+lines',
            'x': [0, vec[0]],
            'y': [0, vec[1]],
            'name': f'Axis {i}'
        } for i, vec in enumerate(cell_xy)]

    def _cell_trace2D(self, cell=None, xaxis="x", yaxis="y", color=None, filled=False, **kwargs):

        if cell is None:
            cell = self.geom.cell

        cell_corners = self._get_cell_corners(cell)
        x, y = self._projected_2Dcoords(xyz=cell_corners, xaxis=xaxis, yaxis=yaxis)

        return {
            'type': 'scatter',
            'mode': 'lines',
            'name': 'Unit cell',
            'x': x,
            'y': y,
            'line': {'color': color},
            'fill': 'toself' if filled else None,
            **kwargs
        }
    #---------------------------------------------------
    #                  3D plotting
    #---------------------------------------------------

    def _plot_geom3D(self, wrap_atom=None, wrap_bond=None, cell='axes'):
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
        
        self._plot_geom(
            self._atom_trace3D, self._bond_trace3D, self._cell_axes_traces3D, self._cell_trace3D,
            wrap_atom, wrap_bond , cell=cell   
        )
            
        forbidden3D_keys = ('title', 'showlegend', 'paper_bgcolor', 'plot_bgcolor',
                          'xaxis_scaleanchor', 'xaxis_scaleratio', 'yaxis_scaleanchor', 'yaxis_scaleratio')
        
        self.layout.scene = {
            'aspectmode': 'data', 
            **{key:val for key, val in self.settings_group("layout").items() if key not in forbidden3D_keys}
        }

    def _default_wrap_atom3D(self, at):

        return (self.geom[at], 0.3), {"name": f'{at+1} ({self.geom.atom[at].tag})'}

    def _default_wrap_bond3D(self, bond):

        return (*self.geom[bond], 15), {}

    def _cell_axes_traces3D(self, cell=None):

        if cell is None:
            cell = self.geom.cell

        return [{
            'type': 'scatter3d',
            'x': [0, vec[0]],
            'y': [0, vec[1]],
            'z': [0, vec[2]],
            'name': f'Axis {i}'
        } for i, vec in enumerate(cell)]
    
    def _cell_trace3D(self, cell=None, color=None, width=2, **kwargs):

        if cell is None:
            cell = self.geom.cell

        x, y, z = self._get_cell_corners(cell).T

        return {
            'type': 'scatter3d',
            'mode': 'lines',
            'name': 'Unit cell',
            'x': x,
            'y': y,
            'z': z,
            'line': {'color': color, 'width': width},
            **kwargs
        }
        
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
        
    def _bond_trace3D(self, xyz1, xyz2, r=15, color="#ccc", name=None, group=None, showlegend=False, **kwargs):
        
        # Drawing cylinders instead of lines would be better, but rendering would be slower
        # We need to give the possibility.
        # Also, the fastest way to draw bonds would be a single scatter trace with just markers
        # (bonds would be drawn as sequences of points, but rendering would be much faster)

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

    _parameters = (

        DropdownInput(key="ndims", name="Dimensions",
            default=3,
            width="s100% m50% l90%",
            params={
                'options': [
                    {'label': '1', 'value': 1},
                    {'label': '2', 'value': 2},
                    {'label': '3', 'value': 3}
                ],
                'isMulti': False,
                'isSearchable': True,
                'isClearable': False
            },
            help='''The dimensionality of the plot'''
        ),

        DropdownInput(key="cell", name="Cell display",
            default="box",
            width="s100% m50% l90%",
            params={
                'options': [
                    {'label': 'False', 'value': False},
                    {'label': 'axes', 'value': 'axes'},
                    {'label': 'box', 'value': 'box'}
                ],
                'isMulti': False,
                'isSearchable': True,
                'isClearable': False
            },
            help='''Specifies how the cell should be rendered. 
            (False: not rendered, 'axes': render axes only, 'box': render a bounding box)'''
        ),

        ProgramaticInput(key="xaxis", name="X axis", default="x"),
        ProgramaticInput(key="yaxis", name="Y axis", default="y")
        
    )

    def _set_data(self):

        ndims = self.setting("ndims")
        cell_rendering = self.setting("cell")

        if ndims == 3:
            self._plot_geom3D(cell=cell_rendering)
        elif ndims == 2:
            xaxis = self.setting("xaxis")
            yaxis = self.setting("yaxis")
            self._plot_geom2D(xaxis=xaxis, yaxis=yaxis, cell=cell_rendering)
            self.update_settings(update_fig=False, xaxis_title=f'Axis {xaxis} (Ang)', yaxis_title=f'Axis {yaxis} (Ang)', no_log=True)
        else:
            raise NotImplementedError

    def _after_get_figure(self):

        ndims = self.setting("ndims")

        if ndims == 2:
            self.layout.yaxis.scaleanchor = "x"
            self.layout.yaxis.scaleratio = 1



