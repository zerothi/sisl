from functools import wraps
from collections import Iterable, defaultdict

import numpy as np

from sisl import Geometry, PeriodicTable, Atom
from ..plot import Plot, entry_point
from ..input_fields import ProgramaticInput, FunctionInput, FloatInput, SwitchInput, DropdownInput, AtomSelect, GeomAxisSelect, \
    FilePathInput, PlotableInput
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
    
    _layout_defaults = {
        'xaxis_showgrid': False,
        'xaxis_zeroline': False,
        'yaxis_showgrid': False,
        'yaxis_zeroline': False,
    }

    @property
    def on_geom(self):
        return BoundGeometry(self.geometry, self)

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
            
    def _after_read(self):

        if self.setting('bonds'):
            self.bonds = self.find_all_bonds(self.geometry)
    
    @staticmethod
    def find_all_bonds(geom):
        
        pt = PeriodicTable()

        bonds = []
        for at in geom:
            neighs = geom.close(at, R=[0.1, 3])[-1]

            for neigh in neighs:
                if pt.radius([geom.atoms[at].Z, geom.atoms[neigh % geom.na].Z]).sum() + 0.15 > np.linalg.norm(geom[neigh] - geom[at]):
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
                axis = self.geometry.cell[i]
        elif isinstance(axis, int):
            axis = self.geometry.cell[axis]
        
        return np.array(axis)
    
    def _get_cell_corners(self, cell=None, unique=False):

        if cell is None:
            cell = self.geometry.cell

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
    
    def _get_atoms_bonds(self, bonds, atom, geom=None, sanitize_atom=True):
        '''
        Gets the bonds where the given atoms are involved
        '''

        if atom is None:
            return bonds

        if sanitize_atom:
            geom = geom or self.geometry
            atom = geom._sanitize_atom(atom)
        
        return [bond for bond in bonds if np.any([at in atom for at in bond])]
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

        x = self._projected_1Dcoords(self.geometry.xyz, axis=coords_axis)
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
            "text": [f'{self.geometry[at]}<br>{at+1} ({self.geometry.atoms[at].tag})' for at in self.geometry],
            "name": "Atoms",
            "color": [self.atom_color(atom.Z) for atom in self.geometry.atoms],
            "size": [self._pt.radius(atom.Z)*16 for atom in self.geometry.atoms]
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
            xyz = self.geometry.xyz

        # Get the directions that these axes represent if the provided input
        # is an axis index
        axis = self._sanitize_axis(axis)

        return xyz.dot(axis)/np.linalg.norm(axis)

    #---------------------------------------------------
    #                  2D plotting
    #---------------------------------------------------

    def _plot_geom2D(self, xaxis="x", yaxis="y", atom=None , show_bonds=True, bind_bonds_to_ats=True,
        bonds_together=True, points_per_bond=5,
        cell='box', wrap_atoms=None, wrap_bond=None):
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
        atom: array-like of int, optional
            the indices of the atoms that you want to plot
        show_bonds: boolean, optional
            whether bonds should be plotted.
        bind_bonds_to_ats: boolean, optional
            whether only the bonds that belong to an atom that is present should be displayed.
            If False, all bonds are displayed regardless of the `atom` parameter.
        bonds_together: boolean, optional
            If set to True, it draws all bonds in one trace, which may be faster for rendering.
            The only limitation that it has is that you can't set individual widths.

            If you provide variable color and/or size for the bonds, bonds will be drawn as dots
            (if you use enough points per bond it almost looks like a line). If you don't like this, use individual
            bonds instead, but then note that you can not share a colorscale between bonds. This indirectly means that you 
            can not provide the color as a number, so you will need to calculate the colors yourself if you want
            a colorscale-like behavior.
        points_per_bond: int, optional
            If `bonds_together` is True and you provide a variable color or size (using `wrap_bonds`), this is
            the number of points that are used for each bond. See `bonds_together` for more info.
        cell: {False, "box", "axes"}, optional
            determines how the unit cell is represented.
        wrap_atoms: function, optional
            function that recieves the 2D coordinates and returns
            the args (array-like) and kwargs (dict) that go into self._atoms_scatter_trace2D()

            If not provided, self._default_wrap_atoms2D will be used.
            wrap_atom: function, optional
            function that recieves the index of an atom and returns
            the args (array-like) and kwargs (dict) that go into self._atom_trace3D()

            If not provided, self._default_wrap_atom3D will be used.
        wrap_bond: function, optional
            function that recieves "a bond" (list of 2 atom indices) and its coordinates ((x1,y1), (x2, y2)).
            It should return the args (array-like) and kwargs (dict) that go into `self._bond_trace2D()`

            If not provided, self._default_wrap_bond2D will be used.
        cell: {'axes', 'box', False}, optional
            defines how the unit cell is drawn
        '''

        wrap_atoms = wrap_atoms or self._default_wrap_atoms2D
        wrap_bond = wrap_bond or self._default_wrap_bonds2D

        if atom is not None:
            atom = self.geometry._sanitize_atom(atom)

        xy = self._projected_2Dcoords(self.geometry[atom], xaxis=xaxis, yaxis=yaxis)
        traces = []

        # Add bonds
        if show_bonds:
            # Define the actual bonds that we are going to draw depending on which
            # atoms are requested
            bonds = self.bonds
            if bind_bonds_to_ats:
                bonds = self._get_atoms_bonds(bonds, atom, sanitize_atom=False)

            if bonds_together:
                
                bonds_xyz = np.array([self.geometry[bond] for bond in bonds])
                xys = self._projected_2Dcoords(bonds_xyz, xaxis=xaxis, yaxis=yaxis)

                # By reshaping we get the following: First axis -> bond (length: number of bonds),
                # Second axis -> atoms in the bond (length 2), Third axis -> coordinate (x, y)
                xys = xys.transpose((1,2,0))

                # Try to get the bonds colors (It might be that the user is not setting them)
                bondsinfo = [wrap_bond(bond, xy) for bond, xy in zip(bonds, xys)]

                bondsprops = defaultdict(list)
                for bondinfo in bondsinfo:
                    if "color" in bondinfo[1]:
                        bondsprops["bonds_color"].append(bondinfo[1]["color"])
                    if "name" in bondinfo[1]:
                        bondsprops["bonds_labels"].append(bondinfo[1]["name"])

                bonds_trace = self._bonds_scatter_trace2D(xys, points_per_bond=points_per_bond, **bondsprops)
                traces.append(bonds_trace)

            else:
                for bond in self.bonds:
                    xys = self._projected_2Dcoords(self.geometry[bond], xaxis=xaxis, yaxis=yaxis)
                    bond_args, bond_kwargs = wrap_bond(bond, xys.T)
                    trace = self._bond_trace2D(*bond_args, **bond_kwargs)
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
            xyz = self.geometry.xyz

        # Get the directions that these axes represent if the provided input
        # is an axis index
        xaxis = self._sanitize_axis(xaxis)
        yaxis = self._sanitize_axis(yaxis)

        return np.array([xyz.dot(ax)/np.linalg.norm(ax) for ax in (xaxis, yaxis)])

    def _atom_circle_trace2D(self, xyz, r):

        raise NotImplementedError

    def _atoms_scatter_trace2D(self, xyz=None, xaxis="x", yaxis="y", color="gray", size=10, name='atoms', text=None, group=None, showlegend=True, **kwargs):

        if xyz is None:
            xyz = self.geometry.xyz

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

    def _bonds_scatter_trace2D(self, xys, points_per_bond=5, force_bonds_as_points=False, 
        bonds_color='#ccc', bonds_size=3, bonds_labels=None,
        coloraxis="coloraxis", name='bonds', group=None, showlegend=True, **kwargs):
        '''
        Cheaper than _bond_trace2D because it draws all bonds in a single trace.

        It is also more flexible, since it allows providing bond colors as floats that all
        relate to the same colorscale.

        However, the bonds are represented as dots between the two atoms (if you use enough
        points per bond it almost looks like a line).
        '''

        # Check if we need to build the markers_properties from atoms_* arguments
        if isinstance(bonds_color, Iterable) and not isinstance(bonds_color, str):
            bonds_color = np.repeat(bonds_color, points_per_bond)
            single_color = False
        else:
            single_color = True

        if isinstance(bonds_size, Iterable):
            bonds_size = np.repeat(bonds_size, points_per_bond)
            single_size = False
        else:
            single_size = True

        x = []
        y = []
        text = []
        if single_color and single_size and not force_bonds_as_points:
            # Then we can display this trace as lines! :)
            for i, ((x1, y1), (x2, y2)) in enumerate(xys):

                x = [*x, x1, x2, None]
                y = [*y, y1, y2, None]

                if bonds_labels:
                    text = np.repeat(bonds_labels, 3)
                
            
            mode = 'markers+lines'

        else:
            # Otherwise we will need to draw points in between atoms
            # representing the bonds
            for i, ((x1, y1), (x2, y2)) in enumerate(xys):

                x = [*x, *np.linspace(x1, x2, points_per_bond)]
                y = [*y, *np.linspace(y1, y2, points_per_bond)]
            
            mode = 'markers'
            if bonds_labels:
                text = np.repeat(bonds_labels, points_per_bond)

        trace = {
            'type': 'scatter',
            'mode': mode,
            'name': name,
            'x': x,'y': y,
            'marker': {'color': bonds_color, 'size': bonds_size, 'coloraxis': coloraxis},
            'text': text if len(text) != 0 else None,
            'hoverinfo': 'text',
            'legendgroup': group,
            'showlegend': showlegend,
            **kwargs
        }

        return trace

    def _cell_axes_traces2D(self, cell=None, xaxis="x", yaxis="y"):

        if cell is None:
            cell = self.geometry.cell

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
            cell = self.geometry.cell

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
    
    def _default_wrap_bonds2D(self, bond, xys):

        return (*xys, ) , {}

    #---------------------------------------------------
    #                  3D plotting
    #---------------------------------------------------

    def _plot_geom3D(self, wrap_atom=None, wrap_bond=None, cell='box', 
        atom=None, bind_bonds_to_ats=True, atom_vertices=20, show_bonds=True, cheap_bonds=True, cheap_atoms=False, atom_size_factor=40,
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
            atom = self.geometry._sanitize_atom(atom)

        # Draw bonds
        if show_bonds:
            bond_traces = []

            # Define the actual bonds that we are going to draw depending on which
            # atoms are requested
            bonds = self.bonds
            if bind_bonds_to_ats:
                bonds = self._get_atoms_bonds(bonds, atom, sanitize_atom=False)
                
            if cheap_bonds:
                # Draw all bonds in the same trace, also drawing the atoms if requested
                atomsprops = {}

                if cheap_atoms:
                    atomsinfo = [wrap_atom(at) for at in self.geometry]

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
                    
                bond_traces = self._bonds_trace3D(bonds, self.geometry, atoms=cheap_atoms,**bondsprops, **atomsprops, **cheap_bonds_kwargs)
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
            ats = atom if atom is not None else self.geometry
            for i, at in enumerate(ats):
                trace_args, trace_kwargs = wrap_atom(at)
                atom_traces.append(self._atom_trace3D(*trace_args, **{"vertices": atom_vertices, "legendgroup": "atoms", "showlegend": i==0, **trace_kwargs} ))
            self.add_traces(atom_traces)

        # Draw unit cell
        if cell == "axes":
            self.add_traces(self._cell_axes_traces3D())
        elif cell == "box":
            self.add_trace(self._cell_trace3D())
        
        self.layout.scene.aspectmode = 'data'

    def _default_wrap_atom3D(self, at):

        return (self.geometry[at], ), {
            "name": f'{at+1} ({self.geometry.atoms[at].tag})',
            "color": self.atom_color(self.geometry.atoms[at].Z),
            "r": self._pt.radius(self.geometry.atoms[at].Z)*0.6
        }

    def _default_wrap_bond3D(self, bond):

        return (*self.geometry[bond], 15), {}

    def _cell_axes_traces3D(self, cell=None):

        if cell is None:
            cell = self.geometry.cell

        return [{
            'type': 'scatter3d',
            'x': [0, vec[0]],
            'y': [0, vec[1]],
            'z': [0, vec[2]],
            'name': f'Axis {i}'
        } for i, vec in enumerate(cell)]
    
    def _cell_trace3D(self, cell=None, color=None, width=2, **kwargs):

        if cell is None:
            cell = self.geometry.cell

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
        # If only bonds are in this trace, we will 
        if not name:
            name = 'Bonds and atoms' if atoms else 'Bonds'

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
            'showlegend': True,
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
    '''
    Versatile representation of geometries.

    Parameters
    -------------
    axes: None, optional
        The axis along which you want to see the geometry.              You
        can provide as many axes as dimensions you want for your plot.
        Note that the order is important and will result in setting the plot
        axes diferently.             For 2D and 1D representations, you can
        pass an arbitrary direction as an axis (array of shape (3,))
    1d_dataaxis: None, optional
        If you want a 1d representation, you can provide a data axis.
        It should be a function that receives the 1d coordinate of each atom
        and             returns it's "data-coordinate", which will be in the
        y axis of the plot.             If not provided, the y axis will be
        all 0.
    cell: None, optional
        Specifies how the cell should be rendered.              (False: not
        rendered, 'axes': render axes only, 'box': render a bounding box)
    atom: None, optional
        The atoms that are going to be displayed in the plot.
        This also has an impact on bonds (see the `bind_bonds_to_ats` and
        `show_atoms` parameters).             If set to None, all atoms are
        displayed
    bind_bonds_to_ats: bool, optional
        whether only the bonds that belong to an atom that is present should
        be displayed.             If False, all bonds are displayed
        regardless of the `atom` parameter
    show_atoms: bool, optional
        If set to False, it will not display atoms.              Basically
        this is a shortcut for `atom = [], bind_bonds_to_ats=False`.
        Therefore, it will override these two parameters.
    geom: None, optional
    
    geom_file: str, optional
    
    bonds: bool, optional
    
    reading_order: None, optional
        Order in which the plot tries to read the data it needs.
    root_fdf: str, optional
        Path to the fdf file that is the 'parent' of the results.
    results_path: str, optional
        Directory where the files with the simulations results are
        located. This path has to be relative to the root fdf.
    '''

    _plot_type = "Geometry"

    _parameters = (

        PlotableInput(key='geometry', name="Geometry",
            dtype=Geometry,
            default=None,
        ),

        FilePathInput(key="geom_file", name="Geometry file",
            group="dataread",
            default=None
        ),

        SwitchInput(key='bonds', name='Show bonds',
            default=True,
        ),

        GeomAxisSelect(
            key="axes", name="Axes to display",
            default=["x", "y", "z"],
            help='''The axis along which you want to see the geometry. 
            You can provide as many axes as dimensions you want for your plot.
            Note that the order is important and will result in setting the plot axes diferently.
            For 2D and 1D representations, you can pass an arbitrary direction as an axis (array of shape (3,))'''
        ),

        FunctionInput(
            key="1d_dataaxis", name="1d data axis",
            default=None,
            help='''If you want a 1d representation, you can provide a data axis.
            It should be a function that receives the 1d coordinate of each atom and
            returns it's "data-coordinate", which will be in the y axis of the plot.
            If not provided, the y axis will be all 0.
            '''
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

    @entry_point('geometry')
    def _read_nosource(self):
        self.geometry = self.setting('geometry') or getattr(self, "geometry", None)
        
        if self.geometry is None:
            raise Exception("No geometry has been provided.")

    @entry_point('geom_file')
    def _read_siesta_output(self):

        geom_file = self.setting("geom_file") or self.setting("root_fdf")

        self.geometry = self.get_sile(geom_file).read_geometry()

    def _after_read(self):

        BaseGeometryPlot._after_read(self)

        self.get_param("atom").update_options(self.geometry)

    def _set_data(self):

        cell_rendering = self.setting("cell")
        bonds = self.setting('bonds')
        axes = self.setting("axes")
        ndims = len(axes)
        if self.setting("show_atoms") == False:
            atom = []
            bind_bonds_to_ats = False
        else:
            atom = self.setting("atom")
            bind_bonds_to_ats = self.setting("bind_bonds_to_ats")
        
        common_kwargs = {'cell': cell_rendering, 'show_bonds': bonds, 'atom': atom, 'bind_bonds_to_ats': bind_bonds_to_ats}

        if ndims == 3:
            self._plot_geom3D(**common_kwargs)
        elif ndims == 2:
            xaxis, yaxis = axes
            self._plot_geom2D(xaxis=xaxis, yaxis=yaxis, **common_kwargs)
            self.update_layout(xaxis_title=f'Axis {xaxis} [Ang]', yaxis_title=f'Axis {yaxis} [Ang]')
        elif ndims == 1:
            coords_axis = axes[0]
            data_axis = self.setting("1d_dataaxis")
            self._plot_geom1D(coords_axis=coords_axis, data_axis=data_axis )

    def _after_get_figure(self):

        ndims = len(self.setting("axes"))

        if ndims == 2:
            self.layout.yaxis.scaleanchor = "x"
            self.layout.yaxis.scaleratio = 1



