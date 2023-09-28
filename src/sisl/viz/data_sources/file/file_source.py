from pathlib import Path

import sisl

from ..data_source import DataSource


class FileData(DataSource):
    """ Generic data source for reading data from a file.
    
    The aim of this class is twofold:
      - Standarize the way data sources read files.
      - Provide automatic updating features when the read files are updated.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._files_to_read = []

    def follow_file(self, path):
        self._files_to_read.append(Path(path).resolve())
    
    def get_sile(self, path, **kwargs):
        """ A wrapper around get_sile so that the reading of the file is registered"""
        self.follow_file(path)
        return sisl.get_sile(path, **kwargs)

    def function(self, **kwargs):
        if isinstance(kwargs.get('path'), sisl.io.BaseSile):
            kwargs['path'] = kwargs['path'].file
        return self.get_sile(**kwargs)