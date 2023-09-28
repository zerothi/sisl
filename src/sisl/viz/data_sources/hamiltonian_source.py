from pathlib import Path

import sisl

from .data_source import DataSource
from .file.siesta import FileData


class HamiltonianDataSource(DataSource):

    def __init__(self, H=None, kwargs={}):
        super().__init__(H=H, kwargs=kwargs)

    def get_hamiltonian(self, H, **kwargs):
        """ Setup the Hamiltonian object.
        
        Parameters
        ----------
        H : sisl.Hamiltonian
            The Hamiltonian object to be setup.
        """

        if isinstance(H, (str, Path)):
            H = FileData(path=H)
        if isinstance(H, (sisl.io.BaseSile)):
            H = H.read_hamiltonian(**kwargs)

        if H is None:
            raise ValueError("No hamiltonian found.")

        return H
    
    def function(self, H, kwargs):
        return self.get_hamiltonian(H=H, **kwargs)
