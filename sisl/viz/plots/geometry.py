from functools import wraps
from collections import Iterable, defaultdict

import numpy as np

from sisl import Geometry, PeriodicTable, Atom
from sisl.viz import Plot
from sisl.viz.input_fields import ProgramaticInput, FloatInput, SwitchInput, DropdownInput, AtomSelect, FilePathInput
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

    # Colors of the atoms following CPK rules
    _atoms_colors = {
        "H" :"#ccc", # Should be white but the default background is white
        "O" :"Red",
        "Cl" :"Green",
        "N" : "blue",
        "C" : "Grey",
        "S" :"Yellow",
        "P" :"Orange",
        "else": "pink"
    }

    _pt = PeriodicTable()
    
    _parameters = (
        ProgramaticInput(key="geom", name="Geometry",
            default=None
        ),

        FilePathInput(key="geom_file", name="Geometry file",
            group="dataread",
            default=None
        ),

        SwitchInput(key='bonds', name='Show bonds',
            default=True,
        ),
    )
    
    _layout_defaults = {
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
    def _sphere(center=[0,0,0], r=1, vertices=10):
        phi, theta = np.mgrid[0.0:np.pi:complex(0, vertices), 0.0:2.0*np.pi:complex(0, vertices)]
        x = center[0] + r*np.sin(phi)*np.cos(theta)
        y = center[1] + r*np.sin(phi)*np.sin(theta)
        z = center[2] + r*np.cos(phi)
    
        return {'x': x,'y': y,'z': z}
    
    @classmethod
    def atom_color(cls, atom):

        symb = Atom(atom).symbol

        return cls._atoms_colors.get(symb, cls._atoms_colors["else"])
    
    def _read_nosource(self):
        self.geom = self.setting("geom") or getattr(self, "geom", None)
        
        if self.geom is None:
            raise Exception("No geometry has been provided.")

    def _read_siesta_output(self):

        geom_file = self.setting("geom_file") or self.setting("root_fdf")

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
    #                  1D plotting
    #---------------------------------------------------

    def _plot_geom1D(self, coords_axis="x", data_axis=None, wrap_atoms=None, **kwargs):
        '''
        Returns a 1D representation of the plot's geometry.

        Parameters
        -----------
        coords_axis:  {0,1,2, "x", "y", "z", "a", "b", "c"} or array-like of shape 3, optional
            the axis onto which all the atoms are projected.
        data_axis: function, optional
            function that takes the projected 1D coordinates and returns the coordinates for the other axis.
            If not provided, the other axis will just be 0 for all points.
        wrap_atoms: function, optional
            function that takes the 2D positions of the atoms in the plot and returns a tuple of (args, kwargs),
            that are passed to self._atoms_scatter_trace2D.
            If not provided self._default_wrap_atoms is used.
        **kwargs: 
            passed directly to the atoms scatter trace
        '''

        wrap_atoms = wrap_atoms or self._default_wrap_atoms1D
        traces = []

        x = self._projected_1Dcoords(self.geom.xyz, axis=coords_axis)
        if not callable(data_axis):
            def data_axis(x): 
                return np.zeros(x.shape[0])
        y = np.array(data_axis(x))
        xy = np.array([x, y])

        atoms_args, atoms_kwargs = wrap_atoms(xy)
        atoms_kwargs = {**atoms_kwargs, **kwargs}

        traces.append(
            self._atoms_scatter_trace2D(*atoms_args, **atoms_kwargs)
        )

        self.add_traces(traces)

    def _default_wrap_atoms1D(self, xy):

        return (xy, ), {
            "text": [f'{self.geom[at]}<br>{at+1} ({self.geom.atom[at].tag})' for at in self.geom],
            "name": "Atoms",
            "color": [self.atom_color(atom.Z) for atom in self.geom.atoms],
            "size": [self._pt.radius(atom.Z)*16 for atom in self.geom.atoms]
        }

    def _projected_1Dcoords(self, xyz=None, axis="x"):
        '''
        Moves the 3D positions of the atoms to a 2D supspace.

        In this way, we can plot the structure from the "point of view" that we want.

        Parameters
        ------------
        xyz: array-like of shape (natoms, 3), optional
            the 3D coordinates that we want to project.
            otherwise 
        axis: {0,1,2, "x", "y", "z", "a", "b", "c"} or array-like of shape 3, optional
            the direction to be displayed along the X axis. 
            If it's an int, it will interpreted as the index of the cell axis.

        Returns
        ----------
        np.ndarray of shape (natoms, )
            the 2D coordinates of the geometry, with all positions projected into the plane
            defined by xaxis and yaxis.
        '''
        if xyz is None:
            xyz = self.geom.xyz

        # Get the directions that these axes represent if the provided input
        # is an axis index
        axis = self._sanitize_axis(axis)

        return xyz.dot(axis)/np.linalg.norm(axis)

    #---------------------------------------------------
    #                  2D plotting
    #---------------------------------------------------

    def _plot_geom2D(self, xaxis="x", yaxis="y", bonds=True, cell='axes', wrap_atoms=None):
        '''
        Returns a 2D representation of the plot's geometry.

        Parameters
        -----------
        xaxis: {0,1,2, "x", "y", "z", "a", "b", "c"} or array-like of shape 3, optional
            the direction to be displayed along the X axis. 
            If it's an int, it will interpreted as the index of the cell axis.
        yaxis: {0,1,2, "x", "y", "z", "a", "b", "c"} or array-like of shape 3, optional
            the direction to be displayed along the X axis. 
            If it's an int, it will interpreted as the index of the cell axis.
        bonds: boolean, optional
            whether bonds should be plotted.
        cell: {False, "box", "axes"}, optional
            determines how the unit cell is represented.
        wrap_atoms: function, optional
            function that recieves the 2D coordinates and returns
            the args (array-like) and kwargs (dict) that go into self._atoms_scatter_trace2D()

            If not provided, self._default_wrap_atoms2D will be used.
        '''

        wrap_atoms = wrap_atoms or self._default_wrap_atoms2D

        xy = self._projected_2Dcoords(self.geom.xyz, xaxis=xaxis, yaxis=yaxis)
        traces = []

        # Add bonds
        if bonds:
            for bond in self.bonds:
                xys = self._projected_2Dcoords(self.geom[bond], xaxis=xaxis, yaxis=yaxis)
                trace = self._bond_trace2D(*xys.T)
                traces.append(trace)

        # Add atoms
        atoms_args, atoms_kwargs = wrap_atoms(xy)

        traces.append(
            self._atoms_scatter_trace2D(*atoms_args, **atoms_kwargs)
        )

        #Draw cell
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
    
    def _default_wrap_atoms2D(self, xy):

        return self._default_wrap_atoms1D(xy)

    #---------------------------------------------------
    #                  3D plotting
    #---------------------------------------------------

    def _plot_geom3D(self, wrap_atom=None, wrap_bond=None, cell='axes', 
        atom=None, bind_bonds_to_ats=True, atom_vertices=20, cheap_bonds=True, cheap_atoms=False, atom_size_factor=40,
        cheap_bonds_kwargs={}):
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
        cell: {'axes', 'box', False}, optional
            defines how the unit cell is drawn
        atom: array-like of int, optional
            the indices of the atoms that you want to plot
        bind_bonds_to_ats: boolean, optional
            whether only the bonds that belong to an atom that is present should be displayed.
            If False, all bonds are displayed regardless of the `atom` parameter
        atom_vertices: int
            the "definition" of the atom sphere, if not in cheap mode. The more vertices, the more defined the sphere
            will be. However, it will also be more expensive to render.
        cheap_bonds: boolean, optional
            If set to True, it draws all in one trace, which results in a dramatically faster rendering.
            The only limitation that it has is that you can't set individual widths.
        cheap_atoms: boolean, optional
            Whether atoms are drawn in a cheap way (all in one scatter trace). 
            If `False`, each atom is drawn individually as a sphere. It's more expensive, but by doing
            this you avoid variable size problems (and looks better).
        atom_size_factor: float, optional
            in cheap mode, the factor by which the atom sizes will be multiplied.
        cheap_bonds_kwargs: dict, optional
            dict that is passed directly as keyword arguments to `self._bonds_trace3D`.
        '''

        wrap_atom = wrap_atom or self._default_wrap_atom3D
        wrap_bond = wrap_bond or self._default_wrap_bond3D

        if atom is not None:
            atom = self.geom._sanitize_atom(atom)

        # Draw bonds
        if self.setting('bonds'):
            bond_traces = []

            # Define the actual bonds that we are going to draw depending on which
            # atoms are requested
            bonds = self.bonds
            if atom is not None and bind_bonds_to_ats:
                bonds = [bond for bond in bonds if np.any([at in atom for at in bond])]
                
            if cheap_bonds:
                # Draw all bonds in the same trace, also drawing the atoms if requested
                atomsprops = {}

                if cheap_atoms:
                    atomsinfo = [wrap_atom(at) for at in self.geom]

                    atomsprops = {"atoms_color": [], "atoms_size": []}
                    for atominfo in atomsinfo:
                        atomsprops["atoms_color"].append(atominfo[1]["color"])
                        atomsprops["atoms_size"].append(atominfo[1]["r"]*atom_size_factor)
                
                # Try to get the bonds colors (It might be that the user is not setting them)
                bondsinfo = [wrap_bond(bond) for bond in bonds]

                bondsprops = defaultdict(list)
                for bondinfo in bondsinfo:
                    if "color" in bondinfo[1]:
                        bondsprops["bonds_color"].append(bondinfo[1]["color"])
                    if "name" in bondinfo[1]:
                        bondsprops["bonds_labels"].append(bondinfo[1]["name"])
                    
                bond_traces = self._bonds_trace3D(bonds, self.geom, atoms=cheap_atoms,**bondsprops, **atomsprops, **cheap_bonds_kwargs)
            else:
                # Draw each bond individually, allows styling each bond differently
                for bond in self.bonds:
                    trace_args, trace_kwargs = wrap_bond(bond)
                    bond_traces.append(self._bond_trace3D(
                        *trace_args, **trace_kwargs))
                
            self.add_traces(bond_traces)

        # Draw atoms if they are not already drawn 
        if not cheap_atoms:
            atom_traces = []
            ats = atom if atom is not None else self.geom
            for at in ats:
                trace_args, trace_kwargs = wrap_atom(at)
                atom_traces.append(self._atom_trace3D(*trace_args, **trace_kwargs, vertices=atom_vertices))
            self.add_traces(atom_traces)

        # Draw unit cell
        if cell == "axes":
            self.add_traces(self._cell_axes_traces3D())
        elif cell == "box":
            self.add_trace(self._cell_trace3D())
        
        self.layout.scene.aspectmode = 'data'

    def _default_wrap_atom3D(self, at):

        return (self.geom[at], ), {
            "name": f'{at+1} ({self.geom.atom[at].tag})',
            "color": self.atom_color(self.geom.atom[at].Z),
            "r": self._pt.radius(self.geom.atom[at].Z)*0.6
        }

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
        
    def _atom_trace3D(self, xyz, r, color="gray", name=None, group=None, showlegend=False, vertices=10, **kwargs):
        
        trace = {
            'type': 'surface',
            **self._sphere(xyz, r, vertices=vertices),
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

    def _bonds_trace3D(self, bonds, geom_xyz, bonds_width=10, bonds_color='gray', bonds_labels=None,
        atoms=False, atoms_color="blue", atoms_size=None, name=None, coloraxis='coloraxis', legendgroup=None, **kwargs):
        '''
        This method is capable of plotting all the geometry in one 3d trace.

        Parameters
        ----------

        Returns
        ----------
        tuple.
            If bonds_labels are provided, it returns (trace, labels_trace).
            Otherwise, just (trace,)
        '''

        # Check if we need to build the markers_properties from atoms_* arguments
        if atoms and isinstance(atoms_color, Iterable) and not isinstance(atoms_color, str):
            build_marker_color = True
            atoms_color = np.array(atoms_color)
            marker_color = []
        else:
            build_marker_color = False
            marker_color = atoms_color

        if atoms and isinstance(atoms_size, Iterable):
            build_marker_size = True
            atoms_size = np.array(atoms_size)
            marker_size = []
        else:
            build_marker_size = False
            marker_size = atoms_size
        
        # Bond color
        if isinstance(bonds_color, Iterable) and not isinstance(bonds_color, str):
            build_line_color = True
            bonds_color = np.array(bonds_color)
            line_color = []
        else:
            build_line_color = False
            line_color = bonds_color

        x = []; y = []; z = []
        
        for i, bond in enumerate(bonds):

            bond_xyz = geom_xyz[bond]

            x = [*x, *bond_xyz[:, 0], None]
            y = [*y, *bond_xyz[:, 1], None]
            z = [*z, *bond_xyz[:, 2], None]

            if build_marker_color:
                marker_color = [*marker_color, *atoms_color[bond], "white"]
            if build_marker_size:
                marker_size = [*marker_size, *atoms_size[bond], 0]
            if build_line_color:
                line_color = [*line_color, bonds_color[i], bonds_color[i],0]

        if bonds_labels:

            x_labels, y_labels, z_labels = np.array([geom_xyz[bond].mean(axis=0) for bond in bonds]).T 
            labels_trace = {
                'type': 'scatter3d', 'mode': 'markers',
                'x': x_labels, 'y': y_labels, 'z': z_labels,
                'text': bonds_labels, 'hoverinfo': 'text',
                'marker': {'size': bonds_width*3, "color": "rgba(255,255,255,0)"},
                "showlegend": False
            }
            
        trace = {
            'type': 'scatter3d',
            'mode': f'lines{"+markers" if atoms else ""}',
            'name': name,
            'x': x,
            'y': y,
            'z': z,
            'line': {'width': bonds_width, 'color': line_color, 'coloraxis': coloraxis},
            'marker': {'size': marker_size, 'color': marker_color},
            'legendgroup': legendgroup,
            'showlegend': False,
            **kwargs
        }

        return (trace, labels_trace) if bonds_labels else (trace,)

    def _bond_trace3D(self, xyz1, xyz2, r=0.3, color="#ccc", name=None, group=None, showlegend=False, line_kwargs={}, **kwargs):
        
        # Drawing cylinders instead of lines would be better, but rendering would be slower
        # We need to give the possibility.
        # Also, the fastest way to draw bonds would be a single scatter trace with just markers
        # (bonds would be drawn as sequences of points, but rendering would be much faster)

        x, y, z = np.array([xyz1, xyz2]).T
            
        trace = {
            'type': 'scatter3d',
            'mode': 'markers',
            'name': name,
            'x': x,
            'y': y,
            'z': z,
            'line': {'width': r, 'color': color, **line_kwargs},
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

        DropdownInput(key="xaxis", name="X axis",
            default="x",
            params={
                'options': [
                    {'label': ax, 'value': ax} for ax in ["x", "y", "z", 0, 1, 2, "a", "b", "c"]
                ],
                'isMulti': False,
                'isSearchable': True,
                'isClearable': False
            },
        ),

        DropdownInput(key="yaxis", name="Y axis",
            default="y",
            params={
                'options': [
                    {'label': ax, 'value': ax} for ax in ["x", "y", "z", 0, 1, 2, "a", "b", "c"]
                ],
                'isMulti': False,
                'isSearchable': True,
                'isClearable': False
            },
        ),

        AtomSelect(key="atom", name="Atoms to display",
            default=None,
            params={
                "options": [],
                "isSearchable": True,
                "isMulti": True,
                "isClearable": True
            },
            help='''The atoms that are going to be displayed in the plot. 
            This also has an impact on bonds (see the `bind_bonds_to_ats` and `show_atoms` parameters).
            If set to None, all atoms are displayed'''
        ),

        SwitchInput(key="bind_bonds_to_ats", name="Bind bonds to atoms",
            default=True,
            help='''whether only the bonds that belong to an atom that is present should be displayed.
            If False, all bonds are displayed regardless of the `atom` parameter'''
        ),

        SwitchInput(key="show_atoms", name="Show atoms",
            default=True,
            help='''If set to False, it will not display atoms. 
            Basically this is a shortcut for `atom = [], bind_bonds_to_ats=False`.
            Therefore, it will override these two parameters.'''
        )
        
    )

    def _after_read(self):

        BaseGeometryPlot._after_read(self)

        self.get_param("atom").update_options(self.geom)

    def _set_data(self):

        ndims = self.setting("ndims")
        cell_rendering = self.setting("cell")
        if self.setting("show_atoms") == False:
            atom = []
            bind_bonds_to_ats = False
        else:
            atom = self.setting("atom")
            bind_bonds_to_ats = self.setting("bind_bonds_to_ats")
        

        if ndims == 3:
            self._plot_geom3D(cell=cell_rendering, atom=atom, bind_bonds_to_ats=bind_bonds_to_ats)
        elif ndims == 2:
            xaxis = self.setting("xaxis")
            yaxis = self.setting("yaxis")
            self._plot_geom2D(xaxis=xaxis, yaxis=yaxis, cell=cell_rendering)
            self.update_layout(xaxis_title=f'Axis {xaxis} (Ang)', yaxis_title=f'Axis {yaxis} (Ang)')
        elif ndims == 1:
            coords_axis = self.setting("xaxis")
            data_axis = self.setting("yaxis")
            self._plot_geom1D(coords_axis=coords_axis, data_axis=data_axis )

    def _after_get_figure(self):

        ndims = self.setting("ndims")

        if ndims == 2:
            self.layout.yaxis.scaleanchor = "x"
            self.layout.yaxis.scaleratio = 1



